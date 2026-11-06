#!/usr/bin/env python3
"""链路自测：用**极小的人造数据**把 Baseline / TENT / CoTTA 三条评测路径整体跑通。

⚠️ 这不是实验，也不产生任何可用的实验结果。
   它只用 2 个 4 层的小体积（数值是随机噪声 + 任意方块掩膜）验证：
   脚本能否端到端运行、JSON 结构是否正确、以及**两种独立实现是否给出完全一致的 Baseline**。

之所以值得单独做这一步：`eval_tta_seg.py --mode none` 与 `eval_dice.py --raw-dir`
应当算出**逐位相同**的 Dice（同一权重、同一数据、同一指标实现、BN 走 running stats），
若不一致，说明两条路径之间存在隐蔽的预处理或聚合差异——这类问题在真实数据上
极难察觉，只会表现为"复现数字莫名偏低"。

必须在 GPU 上运行：

    python3 tests/test_eval_tta_pipeline.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # repro/code
SELF = ROOT.parent / "runs" / "_selftest"              # repro/runs/_selftest
PY = sys.executable
VOL_SIZE = 64                 # 需 >=64，否则 ResNet 末端特征图缩到 1x1，批统计量会退化
N_SLICES = 4


def make_fake_dataset() -> tuple[Path, Path, Path]:
    """构造 2 个 4 层的小体积与掩膜，返回 (csv, vol_dir, ckpt)。"""
    if SELF.exists():
        shutil.rmtree(SELF)
    vol_dir = SELF / "volumes"
    ann_dir = SELF / "annotations"
    vol_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    rng = np.random.RandomState(0)
    rows = []
    for vi, name in enumerate(["vol_a", "vol_b"]):
        vol = (rng.rand(N_SLICES, VOL_SIZE, VOL_SIZE) * 0.4 + 0.3).astype(np.float32)
        msk = np.zeros((N_SLICES, VOL_SIZE, VOL_SIZE), dtype=np.uint8)
        # 每个切片放一个不同类别的方块，保证三类都非空
        for si in range(N_SLICES):
            label = (si % 3) + 1
            msk[si, 2:8, 2:8] = label
        if vi == 1:
            msk[-1] = 0                      # 最后一层全零 -> 应被"仅标注切片"过滤掉
        np.save(vol_dir / f"{name}.npy", vol)
        np.save(ann_dir / f"{name}.npy", msk)
        rows.append((vol_dir / f"{name}.npy", ann_dir / f"{name}.npy"))

    csv = SELF / "retouch_fake.csv"
    csv.write_text("volume,mask\n" + "\n".join(f"{v},{m}" for v, m in rows) + "\n",
                   encoding="utf-8")

    # 随机权重的分割模型（encoder_weights=None，不联网）
    sys.path.insert(0, str(ROOT / "downstream"))
    import torch
    from seg_lib import build_model

    model = build_model(encoder_weights=None, input_size=VOL_SIZE)
    ckpt = SELF / "seg_fake.pt"
    torch.save({"model": model.state_dict(), "epoch": 0, "input_size": VOL_SIZE,
                "image_range": "0-1", "class_names": ["IRF", "SRF", "PED"],
                "val_dice": {"Mean": 0.0}}, ckpt)
    return csv, vol_dir, ckpt


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"❌ 命令失败: {' '.join(cmd)}")
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(1)


def test_all_modes_run_and_schema_is_stable():
    """三种模式都能跑通，且 JSON 结构一致、Dice 落在合理区间。"""
    csv, vol_dir, ckpt = make_fake_dataset()
    expect_slices = 2 * N_SLICES - 1          # vol_b 的最后一层被过滤

    results = {}
    for mode in ("none", "tent", "cotta"):
        out = SELF / f"tta_{mode}.json"
        run([PY, str(ROOT / "baselines" / "eval_tta_seg.py"),
             "--csv", str(csv), "--ckpt", str(ckpt), "--mode", mode,
             "--out", str(out), "--batch-size", "2", "--input-size", str(VOL_SIZE),
             "--gpu", "0"])
        res = json.loads(out.read_text(encoding="utf-8"))
        results[mode] = res

        assert res["mode"] == f"tta_{mode}", res["mode"]
        assert res["num_slices"] == expect_slices, \
            f"{mode}: 期望 {expect_slices} 个切片，实得 {res['num_slices']}"
        assert len(res["per_slice"]) == expect_slices
        assert res["num_volumes"] == 2, res["num_volumes"]
        for k, v in res["dice"].items():
            assert 0.0 <= v <= 100.0, f"{mode}: {k} 超出 [0,100]: {v}"
        mean_expect = np.mean([res["dice"][k] for k in ("IRF", "SRF", "PED")])
        assert abs(mean_expect - res["dice"]["Mean"]) < 0.06, \
            f"{mode}: Mean 应为三类别简单平均，{mean_expect} vs {res['dice']['Mean']}"
        print(f"  ✓ mode={mode:5s} slices={res['num_slices']} dice={res['dice']}")

    assert results["tent"]["diagnostics"]["num_steps"] > 0, "TENT 未执行任何适配步"
    assert results["cotta"]["diagnostics"]["num_steps"] > 0, "CoTTA 未执行任何适配步"
    print("  ✓ 三种模式均跑通，诊断字段齐备")


def test_baseline_matches_eval_dice():
    """关键交叉验证：--mode none 必须与 eval_dice.py --raw-dir 完全一致。"""
    csv, vol_dir, ckpt = make_fake_dataset()
    out_tta = SELF / "cross_tta.json"
    out_dice = SELF / "cross_dice.json"

    run([PY, str(ROOT / "baselines" / "eval_tta_seg.py"),
         "--csv", str(csv), "--ckpt", str(ckpt), "--mode", "none",
         "--out", str(out_tta), "--batch-size", "2", "--input-size", str(VOL_SIZE),
         "--gpu", "0"])
    run([PY, str(ROOT / "eval" / "eval_dice.py"),
         "--csv", str(csv), "--raw-dir", str(vol_dir), "--ckpt", str(ckpt),
         "--out", str(out_dice), "--batch-size", "16", "--input-size", str(VOL_SIZE),
         "--source-range", "0-1", "--image-range", "0-1", "--gpu", "0"])

    a = json.loads(out_tta.read_text(encoding="utf-8"))
    b = json.loads(out_dice.read_text(encoding="utf-8"))

    assert a["num_slices"] == b["num_slices"], f"切片数不同：{a['num_slices']} vs {b['num_slices']}"
    diff = {k: abs(a["dice"][k] - b["dice"][k]) for k in a["dice"]}
    worst = max(diff.values())
    assert worst < 1e-9, f"两条独立路径的 Baseline 应当逐位一致，最大差异 {worst}"
    print(f"  ✓ 交叉验证 -> eval_tta_seg(none) 与 eval_dice(raw) 完全一致（最大差 {worst:.1e}）")
    print(f"    dice = {a['dice']}")


def main() -> int:
    print("⚠️  链路自测：使用人造小数据，结果不具任何实验意义\n")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        print(f"[{fn.__name__}] {fn.__doc__.splitlines()[0] if fn.__doc__ else ''}")
        try:
            fn()
        except AssertionError as e:
            failed.append((fn.__name__, e))
            print(f"  ✗ 失败: {e}")
    print()
    if failed:
        print(f"❌ {len(failed)}/{len(tests)} 项失败")
        for name, e in failed:
            print(f"   - {name}: {e}")
        return 1
    print(f"✅ 全部 {len(tests)} 项通过（自测数据位于 {SELF}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
