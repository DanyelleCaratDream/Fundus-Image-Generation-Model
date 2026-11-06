#!/usr/bin/env python3
"""seg_lib 的单元测试：用可手算的样例验证 Dice 与值域转换的正确性。

Dice 是本次复现的**唯一核心指标**，一旦实现有偏差，所有对比表都会失真且不会报错。
因此这里对已知答案的样例做断言，可直接运行：

    python3 -m pytest tests/test_seg_lib.py -v
    # 或（无需 pytest）
    python3 tests/test_seg_lib.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downstream"))
from seg_lib import CLASS_NAMES, aggregate_dice, convert_range, dice_per_class  # noqa: E402


def _logit_of(p: float) -> float:
    """把概率 p 转成对应的 logit，便于构造"logits 应判为前景/背景"的样例。"""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def test_perfect_prediction():
    """完全命中：Dice 必须为 1。"""
    target = np.zeros((3, 8, 8), dtype=np.float32)
    target[0, :4, :4] = 1
    target[1, 4:, :4] = 1
    target[2, :, 4:] = 1
    logits = np.where(target > 0, _logit_of(0.99), _logit_of(0.01)).astype(np.float32)
    d = dice_per_class(logits, target, is_logit=True)
    assert np.allclose(d, 1.0), f"完全命中应得 1.0，实得 {d}"
    print("  ✓ 完全命中 -> Dice = 1.000")


def test_known_overlap_handcomputed():
    """手算样例，期望值逐个核算。

    通道 0：GT = 全部 100 像素，Pred = 全部 100 像素 -> Dice = 1.0
    通道 1：GT = 上 5 行（50 像素），Pred = 左 5 列（50 像素），交集 = 5x5 = 25
            -> Dice = 2*25 / (50+50) = 0.5
    通道 2：GT 与 Pred 皆空 -> 记 1.0
    """
    target = np.zeros((3, 10, 10), dtype=np.float32)
    logits = np.full((3, 10, 10), _logit_of(0.01), dtype=np.float32)

    target[0] = 1.0                                  # GT: 100
    logits[0] = _logit_of(0.99)                      # Pred: 100，完全重叠
    expect0 = 1.0

    target[1, 0:5, :] = 1.0                          # GT: 上 5 行 = 50
    logits[1, :, 0:5] = _logit_of(0.99)              # Pred: 左 5 列 = 50，交集 25
    expect1 = 2 * 25 / (50 + 50)

    d = dice_per_class(logits, target, is_logit=True)
    assert abs(d[0] - expect0) < 1e-6, f"通道0 期望 {expect0:.4f}，实得 {d[0]:.4f}"
    assert abs(d[1] - expect1) < 1e-6, f"通道1 期望 {expect1:.4f}，实得 {d[1]:.4f}"
    assert abs(d[2] - 1.0) < 1e-6, f"通道2（空类）期望 1.0，实得 {d[2]:.4f}"
    print(f"  ✓ 手算重叠 -> Dice = [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]"
          f"（期望 [{expect0:.4f}, {expect1:.4f}, 1.0000]）")


def test_empty_class_counts_as_one():
    """GT 与预测皆空 -> 记 1.0（通行约定，避免空类拉低均值）。"""
    target = np.zeros((3, 8, 8), dtype=np.float32)
    logits = np.full((3, 8, 8), _logit_of(0.01), dtype=np.float32)
    d = dice_per_class(logits, target, is_logit=True)
    assert np.allclose(d, 1.0), f"空类应记 1.0，实得 {d}"
    print("  ✓ 空类 -> Dice = 1.000")


def test_false_positive_in_empty_gt():
    """GT 为空但预测有前景 -> 应记 0.0（纯假阳性）。"""
    target = np.zeros((3, 8, 8), dtype=np.float32)
    logits = np.full((3, 8, 8), _logit_of(0.01), dtype=np.float32)
    logits[2, :4, :4] = _logit_of(0.99)
    d = dice_per_class(logits, target, is_logit=True)
    assert abs(d[2] - 0.0) < 1e-6, f"纯假阳性应记 0.0，实得 {d[2]:.4f}"
    assert np.allclose(d[:2], 1.0)
    print("  ✓ 纯假阳性 -> Dice = 0.000")


def test_batch_aggregation_not_first_sample():
    """4D 输入必须跨 batch 合并统计，而不是只取第 0 个样本。

    构造 batch=2：
      样本 0：GT = 上 5 行（50），Pred = 上 5 行（50），交集 50 -> 逐样本 Dice = 1.0
      样本 1：GT = 第 0 行（10），Pred = 下 5 行（50），交集 0   -> 逐样本 Dice = 0.0
    合并统计：交集 50，GT 60，Pred 100 -> Dice = 2*50/160 = 0.625
    逐样本平均：0.5
    两者数值不同，据此可区分实现是否只取了第 0 个样本（那会得到 1.0）。
    """
    target = np.zeros((2, 1, 10, 10), dtype=np.float32)
    logits = np.full((2, 1, 10, 10), _logit_of(0.01), dtype=np.float32)

    target[0, 0, 0:5, :] = 1.0                       # 样本 0 GT: 50
    logits[0, 0, 0:5, :] = _logit_of(0.99)           # 样本 0 Pred: 50，完全重叠

    target[1, 0, 0, :] = 1.0                         # 样本 1 GT: 10
    logits[1, 0, 5:10, :] = _logit_of(0.99)          # 样本 1 Pred: 50，零重叠

    expect_merged = 2 * 50 / (60 + 100)              # 0.625
    d4 = dice_per_class(logits, target, is_logit=True)
    assert abs(d4[0] - expect_merged) < 1e-6, (
        f"4D 输入应跨 batch 合并（期望 {expect_merged:.4f}），实得 {d4[0]:.4f}；"
        "若为 1.0 说明只统计了第 0 个样本"
    )

    per_sample = [dice_per_class(logits[i], target[i], is_logit=True)[0] for i in range(2)]
    assert abs(per_sample[0] - 1.0) < 1e-6, f"样本 0 应得 1.0，实得 {per_sample[0]:.4f}"
    assert abs(per_sample[1] - 0.0) < 1e-6, f"样本 1 应得 0.0，实得 {per_sample[1]:.4f}"
    mean_per_sample = float(np.mean(per_sample))
    assert abs(mean_per_sample - 0.5) < 1e-6
    assert abs(d4[0] - mean_per_sample) > 0.05, "合并统计与逐样本平均应当可区分"
    print(f"  ✓ batch 合并 -> {d4[0]:.4f}；逐样本平均 -> {mean_per_sample:.4f}（语义已区分）")


def test_sigmoid_applied_when_logits_in_unit_range():
    """logits 恰好全部落在 [0,1] 内时也必須过 sigmoid。

    构造全部 logit = 0.2（概率约 0.55，应判为前景）。若误判为概率并直接与 0.5 比较，
    logit=0.2 < 0.5 会被判成背景，Dice 变 0 —— 这正是需要显式 is_logit 的原因。
    """
    target = np.ones((1, 4, 4), dtype=np.float32)
    logits = np.full((1, 4, 4), 0.2, dtype=np.float32)
    d_explicit = dice_per_class(logits, target, is_logit=True)
    assert np.allclose(d_explicit, 1.0), f"logit=0.2 过 sigmoid 后应判为前景，实得 {d_explicit}"

    d_auto = dice_per_class(logits, target)         # 自动判定
    assert np.allclose(d_auto, 1.0), "自动判定也应对非纯 0/1 浮点数组施加 sigmoid"
    print("  ✓ 单位区间内的 logits -> sigmoid 正确施加")


def test_threshold_at_half_probability():
    """概率恰好等于阈值时应判为前景（>= 语义），logit=0 视为前景。"""
    target = np.ones((1, 4, 4), dtype=np.float32)
    logits = np.zeros((1, 4, 4), dtype=np.float32)   # sigmoid(0) = 0.5
    d = dice_per_class(logits, target, is_logit=True)
    assert np.allclose(d, 1.0), f"概率=0.5 应判为前景，实得 {d}"
    print("  ✓ 阈值边界（p=0.5）-> 判为前景")


def test_shape_mismatch_raises():
    """形状不一致必须报错，而不是静默广播或截断。"""
    a = np.zeros((1, 4, 4), dtype=np.float32)
    b = np.zeros((1, 5, 5), dtype=np.float32)
    try:
        dice_per_class(a, b, is_logit=True)
    except ValueError:
        print("  ✓ 形状不一致 -> 正确抛出 ValueError")
        return
    raise AssertionError("形状不一致时未报错")


def test_aggregate_and_std():
    """聚合口径：Mean 为三类别的简单平均（与论文表格口径一致）。"""
    per_slice = [np.array([1.0, 0.5, 0.0]), np.array([0.0, 0.5, 1.0])]
    agg = aggregate_dice(per_slice)
    assert abs(agg["IRF"] - 0.5) < 1e-9
    assert abs(agg["SRF"] - 0.5) < 1e-9
    assert abs(agg["PED"] - 0.5) < 1e-9
    assert abs(agg["Mean"] - 0.5) < 1e-9
    # 逐切片均值口径下，std 反映的是切片间波动（论文 ± 值即此口径）
    assert abs(agg["IRF"] - np.mean([1.0, 0.0])) < 1e-9
    print("  ✓ 聚合 -> Mean 为三类别简单平均，std 为逐切片口径")


def test_convert_range_roundtrip():
    """值域转换必须可逆且端点对齐：-1 -> 0，1 -> 1。"""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = convert_range(x, "-1-1", "0-1")
    assert np.allclose(y, [0.0, 0.5, 1.0]), f"[-1,1]->[0,1] 端口错位: {y}"
    z = convert_range(y, "0-1", "-1-1")
    assert np.allclose(z, x), f"往返不一致: {z}"
    same = convert_range(x, "-1-1", "-1-1")
    assert same is x or np.allclose(same, x)
    print("  ✓ 值域转换 -> 端点对齐且可逆")


def test_class_names_order_locked():
    """类别顺序锁定为 (IRF, SRF, PED)，与论文表格列序一致。"""
    assert CLASS_NAMES == ("IRF", "SRF", "PED"), f"类别顺序被改动: {CLASS_NAMES}"
    print(f"  ✓ 类别顺序 = {CLASS_NAMES}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"运行 {len(tests)} 项单元测试\n")
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
    print(f"✅ 全部 {len(tests)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
