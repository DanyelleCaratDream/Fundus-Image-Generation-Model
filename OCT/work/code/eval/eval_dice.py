#!/usr/bin/env python3
"""评测重建体积的下游分割 DSC（对应论文 Table 1）。

输入
----
* `--recon-dir`：官方仓库推理输出的目录（含 `pred_000.npy`、`pred_001.npy` ...）。
  仓库 `test.py` 以 `enumerate(dataloader)` 的序号命名文件，因此**文件序号 == CSV 行序号**。
  本脚本会校验文件数与形状，若对不上会明确报错而不是静默错配。
* `--csv`：生成该重建所用的设备划分 CSV（如 `retouch_cirrus.csv`），用于拿到 GT 掩膜路径。
* `--ckpt`：`train_seg.py` 产出的分割模型 `best.pt`。
* 或者用 `--raw-dir` 直接评测原始设备图像（论文表格里的 Baseline 行，不做任何适配）。

输出
----
JSON 与一行可粘贴到对比表的文本，含 IRF/SRF/PED/Mean 的均值与标准差。
统计口径与论文一致：**逐 B-scan 计算 Dice，再对切片求均值与标准差**（不是体积级 Dice）。

用法
----
    # 评测重建结果
    python eval_dice.py --csv retouch_cirrus.csv \\
        --recon-dir /runs/xxx/inference/outputs --ckpt /runs/seg/fold0/best.pt \\
        --out /results/ours_cirrus.json

    # 评测未适配的原始图像（Baseline）
    python eval_dice.py --csv retouch_cirrus.csv --raw-dir /data/retouch/cirrus \\
        --ckpt /runs/seg/fold0/best.pt --out /results/baseline_cirrus.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downstream"))
from seg_lib import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_IMAGE_RANGE,
    VALID_RANGES,
    convert_range,
    dice_per_class,
    resize_pair,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval-dice")


def load_seg_model(ckpt_path: Path, device: torch.device):
    """载入下游分割模型（结构与 train_seg.py 严格一致）。"""
    from seg_lib import build_model

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(encoder_weights=None, input_size=ckpt.get("input_size", 224))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    log.info("已载入分割模型: %s（epoch %s，训练时验证 Mean DSC %.4f）",
             ckpt_path, ckpt.get("epoch"), ckpt.get("val_dice", {}).get("Mean", float("nan")))
    return model


def resolve_inputs(args) -> list[tuple[Path, Path]]:
    """构造 (待评测图像体积, GT 掩膜) 配对列表。"""
    df = pd.read_csv(args.csv, dtype=str)

    if args.recon_dir:
        recon_dir = Path(args.recon_dir)
        preds = sorted(recon_dir.glob("pred_*.npy"))
        if not preds:
            raise SystemExit(f"在 {recon_dir} 未找到 pred_*.npy")
        if len(preds) != len(df):
            raise SystemExit(
                f"重建文件数({len(preds)}) 与 CSV 行数({len(df)}) 不一致。\n"
                f"重建目录: {recon_dir}\n"
                "可能原因：推理时用的 CSV 与当前 --csv 不同，或推理被中断。"
                "请核对后再评测，避免序号错配导致指标无意义。"
            )
        pairs = []
        for i, (_, row) in enumerate(df.iterrows()):
            pairs.append((recon_dir / f"pred_{i:03d}.npy", Path(row["mask"])))
        log.info("评测模式: 重建结果（%d 个体积，来源 %s）", len(pairs), recon_dir)
        return pairs

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
        pairs = []
        for _, row in df.iterrows():
            name = Path(row["volume"]).name
            img_path = raw_dir / name
            if not img_path.exists():
                img_path = Path(row["volume"])   # 回退到 CSV 中的原始绝对路径
            pairs.append((img_path, Path(row["mask"])))
        log.info("评测模式: 原始图像 Baseline（%d 个体积，来源 %s）", len(pairs), raw_dir)
        return pairs

    raise SystemExit("必须指定 --recon-dir 或 --raw-dir 之一")


def load_volume_pair(img_path: Path, mask_path: Path, source_range: str,
                     image_range: str) -> tuple[np.ndarray, np.ndarray]:
    """载入一个体积与掩膜，返回 (已转换到下游模型值域的图像, 掩膜)。

    依据仓库源码，原始 `.npy` 体积与推理输出 `pred_*.npy` 都落在 [0, 1]
    （详见 downstream/seg_lib.py 顶部说明）。因此这里不做值域猜测，
    而是由 `--source-range` 显式声明，并对越界情况给出警告。
    """
    if not img_path.exists():
        raise SystemExit(f"图像不存在: {img_path}")
    img = np.load(img_path).astype(np.float32)
    msk = np.load(mask_path).astype(np.uint8)

    lo, hi = float(img.min()), float(img.max())
    if source_range == "0-1" and (lo < -0.02 or hi > 1.02):
        log.warning("%s 声明为 [0,1] 但实测范围 [%.3f, %.3f]，请确认 --source-range 是否正确",
                    img_path.name, lo, hi)
    if source_range == "-1-1" and (lo < -1.02 or hi > 1.02):
        log.warning("%s 声明为 [-1,1] 但实测范围 [%.3f, %.3f]，请确认 --source-range 是否正确",
                    img_path.name, lo, hi)

    img = convert_range(img, source_range, image_range)

    if img.shape != msk.shape:
        raise SystemExit(f"形状不匹配: {img_path.name} {img.shape} vs {mask_path.name} {msk.shape}")
    return img, msk


@torch.no_grad()
def evaluate_volume(model, img, msk, device, input_size, batch_size,
                    annotated_slice_set: set[int] | None = None,
                    annotated_only: bool = True):
    """对一个体积逐 B-scan 预测并计算 Dice。

    返回**逐切片**的 Dice 列表——论文表格是"逐 B-scan 均值±标准差"口径，
    因此必须逐个切片单独计算，不能把整批合并的 Dice 复制给批内每个切片
    （那会让同批所有切片取到同一个值，标准差被人为压小）。

    Args:
        img: 已由 load_volume_pair 转换到下游模型期望值域的体积。
        annotated_slice_set: 显式给出的"被标注切片"集合（取自 RETOUCH 官方清单）。
            RETOUCH 的标注掩膜中，未被标注的切片同样全为 0，与"已标注但无病灶"的切片
            无法区分；若拿到官方标注切片清单，务必通过本参数传入，否则统计口径会有偏差。
        annotated_only: 未提供清单时，退化为"跳过掩膜全零的切片"。
    """
    per_slice: list[dict[str, float]] = []

    for start in range(0, img.shape[0], batch_size):
        stop = min(start + batch_size, img.shape[0])
        batch_imgs, batch_msks = [], []
        for k in range(start, stop):
            if annotated_slice_set is not None:
                if k not in annotated_slice_set:
                    continue
            elif annotated_only and msk[k].sum() == 0:
                continue
            im, mk = resize_pair(img[k], msk[k], input_size)
            batch_imgs.append(im)                      # 值域已在 load_volume_pair 中转好
            batch_msks.append(mk.astype(np.int64))
        if not batch_imgs:
            continue

        x = torch.from_numpy(np.stack(batch_imgs)).float().unsqueeze(1).to(device)
        y = torch.from_numpy(np.stack(batch_msks)).to(device)          # (B, H, W) 整数类别图
        logits = model(x)                                              # (B, 3, H, W)

        for j in range(logits.shape[0]):
            d = dice_per_class(logits[j], y[j], is_logit=True)          # (3,H,W) vs (H,W)
            per_slice.append({CLASS_NAMES[c]: float(d[c]) for c in range(len(CLASS_NAMES))})
    return per_slice


def main() -> int:
    ap = argparse.ArgumentParser(description="评测重建/原始体积的下游分割 DSC")
    ap.add_argument("--csv", required=True, help="设备划分 CSV（与推理时使用的同一份）")
    ap.add_argument("--recon-dir", default=None, help="官方 test.py 输出的 pred_*.npy 目录")
    ap.add_argument("--raw-dir", default=None, help="原始图像目录（Baseline 模式）")
    ap.add_argument("--ckpt", required=True, help="下游分割模型 best.pt")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-range", default=DEFAULT_IMAGE_RANGE, choices=list(VALID_RANGES),
                    help="下游分割模型输入期望的值域（须与训练时一致）")
    ap.add_argument("--source-range", default="0-1", choices=list(VALID_RANGES),
                    help="磁盘上 .npy 的原始值域。仓库约定为 [0-1]（原始体积与重建输出一致）")
    ap.add_argument("--all-slices", action="store_true",
                    help="统计全部 B-scan（默认仅统计被标注的切片，即论文口径）")
    ap.add_argument("--slice-manifest", type=Path, default=None,
                    help="JSON: {\"<体积文件名>\": [切片下标]}，RETOUCH 官方标注切片清单")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log.warning("未检测到 CUDA，将使用 CPU 评测（速度很慢）")

    model = load_seg_model(Path(args.ckpt), device)
    pairs = resolve_inputs(args)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ckpt.get("input_size", args.input_size) != args.input_size:
        log.warning("命令行 input-size=%d 与 checkpoint 记录的 %d 不一致，采用 checkpoint 的值",
                    args.input_size, ckpt["input_size"])
        args.input_size = ckpt["input_size"]

    manifest = None
    if args.slice_manifest:
        manifest = {k: set(int(v) for v in vs)
                    for k, vs in json.loads(Path(args.slice_manifest).read_text()).items()}
        log.info("已载入标注切片清单: %d 个体积", len(manifest))
    elif not args.all_slices:
        log.warning(
            "未提供 --slice-manifest，改为跳过掩膜全零的切片。RETOUCH 中'未标注切片'与"
            "'已标注但无病灶切片'都表现为全零掩膜，两者无法区分；若偏差明显，请补上官方清单。"
        )

    all_slices, per_volume = [], []
    for img_path, mask_path in tqdm(pairs, desc="volumes", ncols=80):
        if not Path(img_path).exists():
            raise SystemExit(f"图像不存在: {img_path}")
        img, msk = load_volume_pair(Path(img_path), mask_path, args.source_range, args.image_range)
        ann_set = manifest.get(Path(img_path).name) if manifest else None
        slices = evaluate_volume(model, img, msk, device, args.input_size,
                                 args.batch_size, annotated_slice_set=ann_set,
                                 annotated_only=not args.all_slices)
        vd = np.mean([[s[c] for c in CLASS_NAMES] for s in slices], axis=0) if slices else np.zeros(3)
        per_volume.append({"volume": Path(img_path).name, "n_slices": len(slices),
                           **{CLASS_NAMES[c]: float(vd[c]) for c in range(3)}})
        all_slices.extend(slices)

    if not all_slices:
        raise SystemExit("没有可用于评测的切片（可能标注全空或被过滤）。")

    agg, std = {}, {}
    for c in CLASS_NAMES:
        vals = np.array([s[c] for s in all_slices])
        agg[c], std[c] = float(vals.mean()), float(vals.std())
    vals = np.array([[s[c] for c in CLASS_NAMES] for s in all_slices]).mean(axis=1)
    agg["Mean"], std["Mean"] = float(vals.mean()), float(vals.std())

    reported = {k: round(v * 100, 1) for k, v in agg.items()}
    reported_std = {k: round(v * 100, 1) for k, v in std.items()}

    result = {
        "mode": "reconstruction" if args.recon_dir else "raw_baseline",
        "csv": str(args.csv),
        "source": str(args.recon_dir or args.raw_dir),
        "ckpt": str(args.ckpt),
        "image_range": args.image_range,
        "source_range": args.source_range,
        "annotated_only": not args.all_slices,
        "slice_manifest": str(args.slice_manifest) if args.slice_manifest else None,
        "num_volumes": len(per_volume),
        "num_slices": len(all_slices),
        "dice": reported,
        "dice_std": reported_std,
        "per_volume": per_volume,
        "per_slice": all_slices,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info("评测完成：%d 个体积 / %d 个 B-scan", len(per_volume), len(all_slices))
    line = " | ".join(f"{k} {reported[k]:5.1f}±{reported_std[k]:4.1f}"
                      for k in (*CLASS_NAMES, "Mean"))
    print(f"\n[{result['mode']}] {line}")
    print(f"结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
