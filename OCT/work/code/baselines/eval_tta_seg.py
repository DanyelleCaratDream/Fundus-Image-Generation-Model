#!/usr/bin/env python3
"""在目标设备上跑 Baseline / TENT / CoTTA 的下游分割 DSC（论文 Table 1 的对照行）。

统一入口，三种模式共用**完全相同**的数据路径、分割权重、切片集合与 Dice 实现：

    --mode none    不做适配，源域模型直接推理  ← 与 eval_dice.py --raw-dir 等价，可互校
    --mode tent    在线熵最小化（只更新归一化层仿射参数）
    --mode cotta   教师-学生 EMA + 随机回退 + 增强平均伪标签

三者输出同为 eval_dice.py 的 JSON 结构，因此可直接汇总进同一张对比表。

重要说明
--------
* 适配对象是**下游分割模型**，输入是目标设备（Cirrus/Topcon）的**原始图像**；
  本脚本不接触任何重建结果。论文表格中 TENT/CoTTA 行的 FID 为 N/A 即因它们不产生图像。
* TENT/CoTTA 会额外把归一化层切换到**测试批统计量**（这是方法本身的一部分），
  而 `--mode none` 保持源域 running stats —— 这正是论文 Baseline 行的定义。
* 目标域没有任何标签参与适配：伪标签全部来自模型自身输出。

用法
----
    python eval_tta_seg.py --csv retouch_cirrus.csv --ckpt runs/seg/fold0/best.pt \\
        --mode tent --out results/tent_cirrus.json --batch-size 16

    python eval_tta_seg.py --csv retouch_cirrus.csv --ckpt runs/seg/fold0/best.pt \\
        --mode cotta --out results/cotta_cirrus.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "downstream"))
from seg_lib import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_IMAGE_RANGE,
    RetouchSegDataset,
    VALID_RANGES,
    build_model,
    dice_per_class,
)
from tta_seg import CottaAdapter, CottaConfig, TentAdapter, TentConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval-tta")


class NoAdapt:
    """不做适配的对照（论文 Baseline 行）：源域模型 + 源域 running stats。"""

    name = "none"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()

    def describe(self) -> dict:
        return {"name": "none"}

    def reset(self) -> None:  # 与其它适配器保持同一接口
        pass

    def adapt(self, x: torch.Tensor) -> dict:
        return {"skipped": True}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(x)


def load_seg_model(ckpt_path: Path, device: torch.device) -> dict:
    """载入下游分割模型，返回 checkpoint 内容（供读取 input_size / image_range）。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(encoder_weights=None, input_size=ckpt.get("input_size", 224))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    log.info("已载入分割模型 %s（epoch %s，训练时验证 Mean DSC %.4f）",
             ckpt_path, ckpt.get("epoch"), ckpt.get("val_dice", {}).get("Mean", float("nan")))
    return ckpt


def build_adapter(mode: str, model: torch.nn.Module, args) -> object:
    """按模式构造适配器。"""
    if mode == "none":
        return NoAdapt(model)
    if mode == "tent":
        cfg = TentConfig(lr=args.tent_lr, momentum=args.tent_momentum,
                         steps=args.tent_steps, scope=args.tent_scope,
                         use_batch_stats=not args.tent_source_stats)
        return TentAdapter(model, cfg)
    if mode == "cotta":
        cfg = CottaConfig(lr=args.cotta_lr, momentum=args.cotta_momentum,
                          steps=args.cotta_steps, ema_alpha=args.cotta_ema,
                          restore_prob=args.cotta_restore, n_aug=args.cotta_n_aug,
                          conf_threshold=args.cotta_conf, queue_size=args.cotta_queue,
                          scope=args.cotta_scope, num_classes=len(CLASS_NAMES),
                          seed=args.seed)
        return CottaAdapter(model, cfg)
    raise SystemExit(f"未知 --mode: {mode!r}")


def fallback_to_source_stats(adapter, batch_n: int) -> bool:
    """批大小为 1 时批统计量退化，临时回退到源域 running stats。

    原因：TENT/CoTTA 用测试批统计量做归一化，而单个样本无法给出有意义的均值/方差；
    当末端特征图缩到 1×1 时（小输入尺寸下会发生），PyTorch 的 BatchNorm 会直接报错。
    丢弃该批次会**静默改变评测切片集合**，因此这里选择回退并如实记录发生次数。
    返回 True 表示调用方需要在本批结束后恢复 use_batch_stats。
    """
    cfg = getattr(adapter, "cfg", None)
    if cfg is None or batch_n > 1 or not getattr(cfg, "use_batch_stats", False):
        return False
    cfg.use_batch_stats = False
    return True


def collate(samples) -> dict:
    """把 SegSample 列表拼成 batch，并带上体积名与切片号便于按体积聚合。"""
    return {
        "image": torch.from_numpy(np.stack([s.image for s in samples])),
        "target": torch.from_numpy(np.stack([s.target for s in samples])),
        "volume": [Path(s.volume).name for s in samples],
        "slice": [int(s.slice_index) for s in samples],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline / TENT / CoTTA 的下游分割 DSC")
    ap.add_argument("--csv", required=True, type=Path, help="目标设备划分 CSV（如 retouch_cirrus.csv）")
    ap.add_argument("--ckpt", required=True, type=Path, help="下游分割模型 best.pt")
    ap.add_argument("--mode", required=True, choices=["none", "tent", "cotta"])
    ap.add_argument("--out", required=True, type=Path, help="输出 JSON 路径")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="适配与推理的批大小。TENT/CoTTA 依赖批统计量，必须 >1")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--image-range", default=DEFAULT_IMAGE_RANGE, choices=list(VALID_RANGES))
    ap.add_argument("--all-slices", action="store_true",
                    help="统计全部 B-scan（默认仅统计被标注的切片）")
    ap.add_argument("--slice-manifest", type=Path, default=None,
                    help='JSON: {"<体积文件名>": [切片下标]}，RETOUCH 官方标注切片清单')
    ap.add_argument("--shuffle", action="store_true",
                    help="打乱测试流（i.i.d. 协议）。默认按 CSV 顺序逐体积流入，更接近真实临床序列")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reset-each-batch", action="store_true",
                    help="episodic 协议：每个批次前把模型恢复到源域初始值")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="默认 0：数据集按体积缓存，多进程会成倍复制内存")
    ap.add_argument("--gpu", type=int, default=0)

    g = ap.add_argument_group("TENT")
    g.add_argument("--tent-lr", type=float, default=1e-3)
    g.add_argument("--tent-momentum", type=float, default=0.9)
    g.add_argument("--tent-steps", type=int, default=1)
    g.add_argument("--tent-scope", default="bn_affine", choices=["bn_affine", "all"])
    g.add_argument("--tent-source-stats", action="store_true",
                   help="TENT 改用源域 running stats（偏离原论文设定，仅作对照）")

    g = ap.add_argument_group("CoTTA")
    g.add_argument("--cotta-lr", type=float, default=1e-3)
    g.add_argument("--cotta-momentum", type=float, default=0.9)
    g.add_argument("--cotta-steps", type=int, default=1)
    g.add_argument("--cotta-ema", type=float, default=0.999)
    g.add_argument("--cotta-restore", type=float, default=0.01)
    g.add_argument("--cotta-n-aug", type=int, default=2)
    g.add_argument("--cotta-conf", type=float, default=0.5)
    g.add_argument("--cotta-queue", type=int, default=100)
    g.add_argument("--cotta-scope", default="all", choices=["bn_affine", "all"])
    args = ap.parse_args()

    if args.mode in ("tent", "cotta") and args.batch_size < 2:
        raise SystemExit("TENT/CoTTA 依赖批统计量，--batch-size 必须 >= 2，否则结果不可信")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA。TENT/CoTTA 的基线评测必须用 GPU（CPU 会跑不动）。")
    device = torch.device(f"cuda:{args.gpu}")

    ckpt = load_seg_model(args.ckpt, device)
    if ckpt.get("input_size") and ckpt["input_size"] != args.input_size:
        log.warning("--input-size=%d 与 checkpoint 记录的 %d 不一致，采用 checkpoint 的值",
                    args.input_size, ckpt["input_size"])
        args.input_size = ckpt["input_size"]
    if ckpt.get("image_range") and ckpt["image_range"] != args.image_range:
        log.warning("--image-range=%s 与 checkpoint 记录的 %s 不一致，采用 checkpoint 的值",
                    args.image_range, ckpt["image_range"])
        args.image_range = ckpt["image_range"]

    manifest = None
    if args.slice_manifest:
        manifest = {k: set(int(v) for v in vs)
                    for k, vs in json.loads(args.slice_manifest.read_text()).items()}
        log.info("已载入标注切片清单：%d 个体积", len(manifest))

    dataset = RetouchSegDataset(
        csv_path=args.csv, input_size=args.input_size, image_range=args.image_range,
        annotated_only=not args.all_slices, annotated_slices=manifest,
    )
    if len(dataset) == 0:
        raise SystemExit("数据集为空，请检查 CSV、数据路径与标注切片清单")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )
    log.info("测试流：%d 个 B-scan / %d 个体积，batch=%d，顺序=%s",
             len(dataset), len(dataset.df), args.batch_size,
             "打乱" if args.shuffle else "逐体积顺序")

    # 适配器需要独立的分割模型实例（CoTTA 内部会 deepcopy 出教师模型）
    model = build_model(encoder_weights=None, input_size=args.input_size)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    adapter = build_adapter(args.mode, model, args)
    log.info("适配配置：%s", adapter.describe())

    all_slices: list[dict] = []
    infos: list[dict] = []
    n_singleton = 0
    t0 = time.time()

    for batch in tqdm(loader, desc=f"{args.mode}", ncols=80):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        if args.reset_each_batch:
            adapter.reset()

        fell_back = fallback_to_source_stats(adapter, int(x.shape[0]))
        if fell_back:
            n_singleton += 1
        try:
            info = adapter.adapt(x)
            if args.mode != "none":
                infos.append(info)
            logits = adapter.predict(x)                  # (B, 3, H, W)
        finally:
            if fell_back:
                adapter.cfg.use_batch_stats = True
        for j in range(logits.shape[0]):
            d = dice_per_class(logits[j], y[j], is_logit=True)
            all_slices.append({
                "volume": batch["volume"][j], "slice": batch["slice"][j],
                **{CLASS_NAMES[c]: float(d[c]) for c in range(len(CLASS_NAMES))},
            })

    elapsed = time.time() - t0
    if not all_slices:
        raise SystemExit("没有可用于评测的切片")

    # ---- 聚合（与 eval_dice.py 同口径：逐切片均值与标准差）----
    agg, std = {}, {}
    for c in CLASS_NAMES:
        vals = np.array([s[c] for s in all_slices])
        agg[c], std[c] = float(vals.mean()), float(vals.std())
    vals = np.array([[s[c] for c in CLASS_NAMES] for s in all_slices]).mean(axis=1)
    agg["Mean"], std["Mean"] = float(vals.mean()), float(vals.std())
    reported = {k: round(v * 100, 1) for k, v in agg.items()}
    reported_std = {k: round(v * 100, 1) for k, v in std.items()}

    vol_names = sorted({s["volume"] for s in all_slices})
    per_volume = []
    for v in vol_names:
        ss = [s for s in all_slices if s["volume"] == v]
        vd = np.mean([[s[c] for c in CLASS_NAMES] for s in ss], axis=0)
        per_volume.append({"volume": v, "n_slices": len(ss),
                           **{CLASS_NAMES[c]: float(vd[c]) for c in range(len(CLASS_NAMES))}})

    diagnostics: dict = {"num_steps": len(infos) if infos else 0}
    if args.mode == "cotta" and infos:
        diagnostics["skipped_steps"] = sum(1 for i in infos if i.get("skipped"))
        diagnostics["selected_class_hist"] = dict(Counter(i["class"] for i in infos
                                                          if not i.get("skipped")))
        losses = [i["loss"] for i in infos if "loss" in i]
        diagnostics["mean_loss"] = float(np.mean(losses)) if losses else None
    if args.mode == "tent" and infos:
        curves = [i["entropy_curve"] for i in infos if i.get("entropy_curve")]
        if curves:
            first = float(np.mean([c[0] for c in curves]))
            last = float(np.mean([c[-1] for c in curves]))
            diagnostics["entropy_step0_mean"] = first
            diagnostics["entropy_step_last_mean"] = last
            diagnostics["entropy_decreased"] = bool(last < first)

    result = {
        "mode": f"tta_{adapter.name}",
        "csv": str(args.csv),
        "ckpt": str(args.ckpt),
        "image_range": args.image_range,
        "input_size": args.input_size,
        "annotated_only": not args.all_slices,
        "slice_manifest": str(args.slice_manifest) if args.slice_manifest else None,
        "batch_size": args.batch_size,
        "shuffle": bool(args.shuffle),
        "reset_each_batch": bool(args.reset_each_batch),
        "num_volumes": len(per_volume),
        "num_slices": len(all_slices),
        "seconds": elapsed,
        "tta_config": adapter.describe(),
        "singleton_batches_using_source_stats": n_singleton,
        "diagnostics": diagnostics,
        "dice": reported,
        "dice_std": reported_std,
        "per_volume": per_volume,
        "per_slice": all_slices,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    line = " | ".join(f"{k} {reported[k]:5.1f}±{reported_std[k]:4.1f}"
                      for k in (*CLASS_NAMES, "Mean"))
    print(f"\n[{result['mode']}] {line}")
    if diagnostics:
        print(f"适配诊断：{diagnostics}")
    print(f"耗时 {elapsed:.1f}s，结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
