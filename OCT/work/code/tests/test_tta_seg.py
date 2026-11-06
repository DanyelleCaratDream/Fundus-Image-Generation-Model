#!/usr/bin/env python3
"""TENT / CoTTA 适配器的单元测试。

这些测试不依赖 RETOUCH 数据，验证的是**适配器语义**而非实验数值：
参数范围是否精确、梯度是否只流向该更新的参数、EMA/随机回退是否按公式执行、
队列与跳过逻辑是否正确。这些性质一旦错，基线对比就失去意义且不会报错。

跑法：

    python3 -m pytest tests/test_tta_seg.py -v
    # 或（无需 pytest）
    python3 tests/test_tta_seg.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from tta_seg import (  # noqa: E402
    CottaAdapter,
    CottaConfig,
    TentAdapter,
    TentConfig,
    affine_params,
    binary_entropy,
    grayscale_augment,
    set_norm_batch_stats,
    set_trainable,
    small_affine,
)


class TinySeg(nn.Module):
    """最小可用的"分割模型"：Conv-BN-ReLU x2 + 1x1 分类头，输出 3 通道 logits。"""

    def __init__(self, ch: int = 8, n_classes: int = 3) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
        )
        self.head = nn.Conv2d(ch, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.f(x))


def _shifted_batch(n: int = 8, size: int = 24, seed: int = 0) -> torch.Tensor:
    """构造一个"目标域"批次：均值/方差都明显偏离标准正态，模拟跨设备偏移。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, size, size, generator=g) * 3.0 + 2.0


# --------------------------------------------------------------------------
# 参数范围与梯度流向
# --------------------------------------------------------------------------
def test_set_trainable_scopes():
    """bn_affine 只选中归一化仿射参数；all 选中全部参数。"""
    m = TinySeg()
    n_affine = sum(p.numel() for p in affine_params(m))
    # 2 个 BatchNorm2d(ch=8)，各含 weight/bias -> 32 个参数
    assert n_affine == 32, f"预期 32 个仿射参数，实得 {n_affine}"

    got = set_trainable(m, "bn_affine")
    assert sum(p.numel() for p in got) == 32, "bn_affine 范围不正确"
    assert all(not p.requires_grad for p in m.f[0].parameters()), "卷积层不应可训练"

    got_all = set_trainable(m, "all")
    assert sum(p.numel() for p in got_all) == sum(p.numel() for p in m.parameters())
    print(f"  ✓ 参数范围 -> bn_affine {n_affine} 个 / all {sum(p.numel() for p in m.parameters())} 个")


def test_grad_only_flows_to_norm_affine():
    """梯度必须只流向归一化仿射参数，否则 TENT 就不再是 TENT。"""
    m = TinySeg()
    set_trainable(m, "bn_affine")
    loss = binary_entropy(m(_shifted_batch(4)))
    loss.backward()

    assert m.f[1].weight.grad is not None, "BN 仿射参数应收到梯度"
    assert m.f[1].bias.grad is not None, "BN 偏置应收到梯度"
    assert m.f[0].weight.grad is None, "卷积权重不应收到梯度"
    assert m.head.weight.grad is None, "分类头不应收到梯度"
    print("  ✓ 梯度流向 -> 仅归一化仿射参数")


# --------------------------------------------------------------------------
# 预测熵
# --------------------------------------------------------------------------
def test_binary_entropy_handcomputed():
    """logit=0 时 p=0.5，各通道二值熵为 ln2，三通道之和为 3·ln2。"""
    logits = torch.zeros(1, 3, 4, 4)
    ent = float(binary_entropy(logits).item())
    expect = 3.0 * math.log(2.0)
    # 实现内部用 eps=1e-6 做数值稳定（log(p+eps)），与解析值存在约 1e-5 量级的系统偏差；
    # 容差取 1e-4 仍足以区分"公式正确"与"漏乘通道数/用错对数底"（后者差异在 0.7 量级）。
    assert abs(ent - expect) < 1e-4, f"期望 {expect:.6f}，实得 {ent:.6f}"
    print(f"  ✓ 预测熵 -> logit=0 时 {ent:.6f}（期望 3·ln2 = {expect:.6f}）")


def test_entropy_scale_is_per_batch_mean():
    """熵对 batch 取平均：同一批复制两份不改变结果（排除误用 sum 的情形）。"""
    torch.manual_seed(99)
    m = TinySeg()                      # 必须用同一个模型，否则两次实例化权重不同、比较无意义
    x = _shifted_batch(4)
    ent1 = float(binary_entropy(m(x)).item())
    ent2 = float(binary_entropy(m(torch.cat([x, x], 0))).item())
    assert abs(ent1 - ent2) < 1e-5, f"应按 batch 取平均：{ent1:.6f} vs {ent2:.6f}"
    print(f"  ✓ 熵归一化 -> 复制批次后仍为 {ent1:.6f}")


# --------------------------------------------------------------------------
# 归一化统计量
# --------------------------------------------------------------------------
def test_batch_stats_does_not_pollute_running_stats():
    """用测试批统计量时，源域 running stats 必须保持不变（否则结果会随评测顺序漂移）。"""
    m = TinySeg()
    rm = m.f[1].running_mean.detach().clone()
    rv = m.f[1].running_var.detach().clone()

    set_norm_batch_stats(m, True)
    _ = m(_shifted_batch(8))
    assert torch.allclose(m.f[1].running_mean, rm), "running_mean 被污染"
    assert torch.allclose(m.f[1].running_var, rv), "running_var 被污染"

    # 且确实切换到了批统计量：与沿用 running stats 的输出应有明显差异
    m2 = TinySeg()
    m2.load_state_dict(m.state_dict())
    set_norm_batch_stats(m2, False)
    x = _shifted_batch(8)
    d = float((m(x) - m2(x)).abs().max().item())
    assert d > 1e-3, f"批统计量与 running stats 应产生不同输出，实测最大差 {d:.2e}"
    print(f"  ✓ 归一化统计量 -> running stats 未污染，批统计量输出差异 {d:.4f}")


# --------------------------------------------------------------------------
# 数据增强
# --------------------------------------------------------------------------
def test_small_affine_identity():
    """零旋转/零平移/缩放 1 应为恒等变换。"""
    x = torch.rand(2, 1, 16, 16)
    y = small_affine(x, angle_deg=0.0, translate=(0.0, 0.0), scale=1.0)
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=1e-4), f"恒等仿射误差过大: {(x - y).abs().max():.2e}"
    print("  ✓ 小幅仿射 -> 恒等参数下还原输入")


def test_grayscale_augment_shape_and_range():
    """增强保持形状，且被夹回输入动态范围（不产生分布外数值）。"""
    x = torch.rand(3, 1, 16, 16)          # [0, 1] 值域
    gen = torch.Generator().manual_seed(7)
    for i in range(6):
        y = grayscale_augment(x, gen)
        assert y.shape == x.shape, f"第 {i} 次形状改变: {y.shape}"
        assert torch.isfinite(y).all(), f"第 {i} 次出现非有限值"
        assert float(y.min()) >= float(x.min()) - 1e-6
        assert float(y.max()) <= float(x.max()) + 1e-6
    print("  ✓ 灰度增强 -> 形状不变且值域被夹回")


def test_grayscale_augment_is_seeded():
    """相同种子产生相同增强序列（保证适配流可复现）。"""
    x = _shifted_batch(2, 16)
    a = [grayscale_augment(x, torch.Generator().manual_seed(11)) for _ in range(2)]
    b = [grayscale_augment(x, torch.Generator().manual_seed(11)) for _ in range(2)]
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
print("  ✓ 灰度增强 -> 同种子可复现")

# --------------------------------------------------------------------------
# TENT
# --------------------------------------------------------------------------
def test_tent_describe():
    """超参导出正常（同时验证 asdict 导入无误）。"""
    d = TentAdapter(TinySeg()).describe()
    assert d["name"] == "tent", d
    assert d["lr"] == 1e-3 and d["scope"] == "bn_affine", d
    print(f"  ✓ TENT 超参 -> {d}")


def test_tent_reduces_entropy():
    """核心性质：熵最小化必须真的把预测熵压下去。"""
    torch.manual_seed(0)
    m = TinySeg()
    ad = TentAdapter(m, TentConfig(lr=0.02, steps=25, scope="bn_affine"))
    x = _shifted_batch(8)

    before = float(binary_entropy(ad.predict(x)).item())
    ad.adapt(x)
    after = float(binary_entropy(ad.predict(x)).item())

    assert after < before, f"熵未下降：{before:.6f} -> {after:.6f}"
    print(f"  ✓ TENT 熵下降 -> {before:.6f} -> {after:.6f}（降幅 {before - after:.6f}）")


def test_tent_only_updates_norm_affine():
    """TENT 的招牌约束：卷积/分类头参数必须一动不动。"""
    torch.manual_seed(1)
    m = TinySeg()
    ad = TentAdapter(m, TentConfig(lr=0.05, steps=5))

    conv = m.f[0].weight.detach().clone()
    head = m.head.weight.detach().clone()
    bn = m.f[1].weight.detach().clone()

    ad.adapt(_shifted_batch(4))

    assert torch.equal(m.f[0].weight, conv), "卷积权重被改动，TENT 范围失守"
    assert torch.equal(m.head.weight, head), "分类头权重被改动，TENT 范围失守"
    assert not torch.allclose(m.f[1].weight, bn), "BN 仿射参数未更新"
    print("  ✓ TENT 更新范围 -> 仅 BN 仿射参数变化，卷积/分类头逐位不变")


def test_tent_reset_restores_source():
    """episodic 模式下 reset 必须把参数恢复到源域初始值。"""
    torch.manual_seed(2)
    m = TinySeg()
    ad = TentAdapter(m, TentConfig(lr=0.05, steps=5))
    bn0 = m.f[1].weight.detach().clone()

    ad.adapt(_shifted_batch(4))
    assert not torch.allclose(m.f[1].weight, bn0), "适配后参数应发生变化"

    ad.reset()
    assert torch.allclose(m.f[1].weight, bn0), "reset 未恢复源域参数"
    print("  ✓ TENT reset -> 参数恢复源域初始值")


# --------------------------------------------------------------------------
# CoTTA
# --------------------------------------------------------------------------
def test_cotta_describe_and_scope():
    """默认更新全部参数；超参导出正常。"""
    m = TinySeg()
    ad = CottaAdapter(m, CottaConfig(scope="all", n_aug=1))
    d = ad.describe()
    assert d["name"] == "cotta" and d["ema_alpha"] == 0.999, d
    assert d["restore_prob"] == 0.01 and d["conf_threshold"] == 0.5, d

    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    assert n_train == n_all, f"scope=all 应训练全部参数：{n_train} vs {n_all}"
    print(f"  ✓ CoTTA 超参 -> scope=all 训练 {n_train} 个参数；{d}")


def test_cotta_teacher_is_independent_copy():
    """教师必须是独立副本，否则学生更新会同时改到教师，EMA 失去意义。"""
    torch.manual_seed(3)
    m = TinySeg()
    ad = CottaAdapter(m, CottaConfig(n_aug=1, restore_prob=0.0))
    with torch.no_grad():
        ad.student.f[0].weight.add_(0.5)
        s_now = ad.student.f[0].weight.detach().clone()
    assert not torch.allclose(ad.teacher.f[0].weight, s_now), "教师与学生共享了参数张量"

    t_before = ad.teacher.f[0].weight.detach().clone()
    ad._ema_update()
    expect = t_before * 0.999 + s_now * 0.001
    err = float((ad.teacher.f[0].weight - expect).abs().max().item())
    assert err < 1e-7, f"EMA 公式不符，最大误差 {err:.2e}"
    print(f"  ✓ CoTTA EMA -> 按 α=0.999 更新，最大误差 {err:.2e}")


def test_cotta_stochastic_restore():
    """随机回退：p=0 不动，p=1 全部回到源域值。"""
    torch.manual_seed(4)
    m = TinySeg()
    ad = CottaAdapter(m, CottaConfig(n_aug=1, restore_prob=0.0, seed=5))
    src = ad.source_params["f.0.weight"].clone()

    with torch.no_grad():
        ad.teacher.f[0].weight.add_(1.0)
    ad._stochastic_restore()
    assert torch.allclose(ad.teacher.f[0].weight, src + 1.0), "p=0 时不应回退"

    ad.cfg.restore_prob = 1.0
    ad._stochastic_restore()
    assert torch.allclose(ad.teacher.f[0].weight, src), "p=1 时应全部回退到源域值"
    print("  ✓ CoTTA 随机回退 -> p=0 无变化，p=1 全部回退")


def test_cotta_adapt_updates_teacher_and_bounds_queue():
    """完整走一遍 adapt：走训练分支、教师被更新、队列长度受 queue_size 约束。"""
    torch.manual_seed(5)
    m = TinySeg()
    ad = CottaAdapter(m, CottaConfig(n_aug=2, queue_size=2, seed=6))
    with torch.no_grad():
        # 让所有像素都稳定超阈值，强制走"有置信伪标签"的训练分支
        ad.teacher.head.weight.fill_(5.0)
        ad.teacher.head.bias.fill_(3.0)

    x = _shifted_batch(4)
    infos = [ad.adapt(x) for _ in range(3)]

    assert all(not i["skipped"] for i in infos), f"应走训练分支：{infos}"
    assert all(math.isfinite(i["loss"]) and i["loss"] >= 0 for i in infos), \
        f"学生更新未执行或损失异常：{infos}"

    cls = infos[0]["class"]
    assert len(ad.queues[cls]) == 2, f"队列长度应被 queue_size=2 约束，实得 {len(ad.queues[cls])}"

    changed = [n for n, p in ad.teacher.named_parameters()
               if not torch.equal(p, ad.source_params[n])]
    assert changed, "教师参数应已更新"
    print(f"  ✓ CoTTA adapt -> 3 步均走训练分支，队列被约束为 2，"
          f"{len(changed)} 个参数张量已更新")


def test_cotta_skips_without_confident_pixels():
    """没有任何像素超阈值时不得更新，也不得污染队列——否则等于拿噪声当监督。"""
    torch.manual_seed(7)
    m = TinySeg()
    ad = CottaAdapter(m, CottaConfig(n_aug=1, seed=8))
    with torch.no_grad():
        ad.teacher.head.weight.zero_()
        ad.teacher.head.bias.fill_(-6.0)      # 恒为低置信背景

    info = ad.adapt(_shifted_batch(4))
    assert info["skipped"] is True, f"应跳过更新：{info}"
    assert info["conf"] < 0, f"空类别置信度应记 -1：{info}"
    assert all(len(q) == 0 for q in ad.queues), "跳过时不应写入队列"
    print(f"  ✓ CoTTA 跳过 -> 全背景批次不更新、不入队（conf={info['conf']:.3f}）")


def test_cotta_reset():
    """episodic 模式下学生与教师都回到源域值，队列清空。"""
    torch.manual_seed(9)
    m = TinySeg()
    ad = CottaAdapter(m, CottaConfig(n_aug=1, seed=10, queue_size=1))
    with torch.no_grad():
        ad.teacher.head.weight.fill_(5.0)
        ad.teacher.head.bias.fill_(3.0)

    ad.adapt(_shifted_batch(4))
    ad.reset()

    for n, p in ad.teacher.named_parameters():
        assert torch.allclose(p, ad.source_params[n]), f"教师未复位：{n}"
    for n, p in ad.student.named_parameters():
        assert torch.allclose(p, ad.source_params[n]), f"学生未复位：{n}"
    assert all(len(q) == 0 for q in ad.queues), "队列未清空"
    print("  ✓ CoTTA reset -> 教师与学生均复位，队列清空")


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
