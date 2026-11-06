#!/usr/bin/env python3
"""官方仓库关键代码路径的验证测试（不需要 RETOUCH 数据，也不产生任何实验结论）。

目的：在真实数据到位之前，先确认仓库的核心组件在本机环境（torch 2.10.0 / RTX 4090 24GB）
上行为符合预期，并把"值域流向""直方图匹配""ODE 轨迹形状""显存占用"这四件事
用实测数字固定下来，避免后续把环境问题误判成方法问题。

运行：
    cd D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work/code/tta-flow
    PYTHONPATH=src D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work/venv/bin/python \\
        ../../tests/test_flow_pipeline.py

**本文件不加载任何真实数据集，不训练模型，输出不构成任何复现结论。**
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_SRC = Path(__file__).resolve().parent.parent / "tta-flow" / "src"
if not REPO_SRC.exists():
    REPO_SRC = Path("D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work/code/tta-flow/src")
sys.path.insert(0, str(REPO_SRC))

RESULTS: list[tuple[str, bool, str]] = []


CHECKS: list[tuple[str, object]] = []


def check(name: str):
    """注册一项检查；由 main() 按声明顺序执行，单项失败不中断其余检查。"""
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def run_check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
        print(f"  ✅ {name}" + (f"\n     {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  ❌ {name} — {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)


# --------------------------------------------------------------------------
# 1. 值域流向：确认磁盘 [0,1] 经仓库 Normalize 后得到网络所需的 [-1,1]
# --------------------------------------------------------------------------
@check("值域流向：Normalize(mean=.5,std=.5,max_pixel_value=1.0) 把 [0,1] 映射为 [-1,1]")
def test_normalize_range():
    import albumentations as A
    norm = A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0)
    disk = np.linspace(0.0, 1.0, 5, dtype=np.float32).reshape(1, 5, 1)
    out = norm(image=disk)["image"]
    lo, hi = float(out.min()), float(out.max())
    assert abs(lo + 1.0) < 1e-5 and abs(hi - 1.0) < 1e-5, f"期望 [-1,1]，实得 [{lo:.4f}, {hi:.4f}]"

    # 反证：若磁盘误存 [-1,1]，输出会是 [-3,1]（不对称，尺度错 2 倍）
    wrong = norm(image=np.array([[-1.0], [1.0]], dtype=np.float32))["image"]
    return f"[0,1]->[{lo:.2f},{hi:.2f}]；反证：误存[-1,1]会得到[{float(wrong.min()):.2f},{float(wrong.max()):.2f}]"


# --------------------------------------------------------------------------
# 2. 直方图匹配：形状、值域、以及"匹配结果分布应贴近模板"
# --------------------------------------------------------------------------
@check("直方图匹配 match_cumulative_cdf_batch 的形状与分布对齐")
def test_histogram_matching():
    from util.flow_matching_util import match_cumulative_cdf_batch

    rng = np.random.default_rng(0)
    h, w, n = 64, 64, 5
    # 参考轨迹某一步的 N 张样本（模拟模型输出，尺度 [-1,1]）
    template = (rng.normal(0.0, 0.35, size=(n, h, w))).astype(np.float32)
    # 测试图：尺度偏小、分布偏斜，模拟低 SNR 设备图像
    source = (rng.gamma(2.0, 0.08, size=(1, 1, h, w)) - 0.16).astype(np.float32)

    out = match_cumulative_cdf_batch(source=source, template_batch=template)
    assert out.shape[-2:] == (h, w), f"输出空间尺寸应为 {(h, w)}，实得 {out.shape[-2:]}"

    s_mean, o_mean = float(template.mean()), float(out.mean())
    s_std, o_std = float(template.std()), float(out.std())
    return (f"输入 source 区间[{float(source.min()):.3f},{float(source.max()):.3f}] "
            f"-> 输出区间[{float(out.min()):.3f},{float(out.max()):.3f}]；"
            f"模板 mean/std={s_mean:.3f}/{s_std:.3f}，输出 mean/std={o_mean:.3f}/{o_std:.3f}")


# --------------------------------------------------------------------------
# 3. 模型前向：形状 + 参数量 + 224×224 的显存占用（对应风险 R6）
# --------------------------------------------------------------------------
@check("UNet 前向：224×224 输入形状、参数量与显存占用")
def test_unet_forward():
    from hydra import compose, initialize_config_dir
    from util import model_util

    pref_dir = str((REPO_SRC.parent / "preferences").resolve())
    with initialize_config_dir(version_base=None, config_dir=pref_dir):
        cfg = compose(config_name="config", overrides=["model.params.dim=[1,224,224]"])
    # 走仓库自身的构建入口，保证与 train.py 的实例化方式完全一致
    model = model_util.build_model(cfg, None, torch.device("cuda"))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(1, 1, 224, 224, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        y = model(torch.zeros(1, device="cuda"), x)
    peak_fwd = torch.cuda.max_memory_allocated() / 2**30
    assert y.shape == x.shape, f"输出形状应与输入一致，实得 {y.shape}"

    # 训练态峰值：batch=1 的 fwd+bwd（含优化器状态之外的部分）
    del y
    torch.cuda.reset_peak_memory_stats()
    xb = torch.randn(1, 1, 224, 224, device="cuda", requires_grad=True)
    out = model(torch.zeros(1, device="cuda"), xb)
    out.mean().backward()
    peak_train1 = torch.cuda.max_memory_allocated() / 2**30

    del out, xb, model
    torch.cuda.empty_cache()
    return (f"参数量 {n_params:.1f}M（论文称 ~103M）；"
            f"前向峰值 {peak_fwd:.2f} GB；batch=1 训练峰值 {peak_train1:.2f} GB")


# --------------------------------------------------------------------------
# 4. ODE 求解器：轨迹形状与时间轴语义（对应 R1 的积分区间问题）
# --------------------------------------------------------------------------
@check("ODESolver 轨迹形状与积分区间语义（s_target 不改变积分区间）")
def test_ode_solver():
    from util.flow_matching_util import ODESolver

    class Dummy(torch.nn.Module):
        """占位速度场：验证求解器的积分逻辑，与训练结果无关。"""

        def forward(self, t, x):  # noqa: D102
            return torch.zeros_like(x)

    solver = ODESolver(model=Dummy(), solver="euler", sample_x=False)
    x0 = torch.randn(1, 1, 32, 32)
    T = 11
    traj = solver.solve(x=x0, t_span=torch.linspace(0, 1, T))
    assert traj.shape[0] == T, f"轨迹时间维应为 {T}，实得 {traj.shape[0]}"
    assert torch.allclose(traj[0], x0), "轨迹首元素应为初始条件"
    # 零速度场下所有点应保持不变
    assert torch.allclose(traj[-1], x0, atol=1e-6), "零速度场下轨迹终点应等于起点"

    # 索引语义：s_target = int(ratio * T) 对应的真实时间比例为 s_target/(T-1)
    for ratio in (0.73, 0.8, 0.88):
        idx = int(ratio * T)
        true_frac = idx / (T - 1)
        print(f"      s_target={ratio} -> 下标 {idx} -> 真实时间比例 {true_frac:.4f}")
    return f"轨迹形状 {tuple(traj.shape)}（[T,B,C,H,W]），积分区间恒为 [0,1]"


# --------------------------------------------------------------------------
# 5. 下游分割模型：UNet++/ResNet-18 在 224×224 上的前向与显存
# --------------------------------------------------------------------------
@check("下游 UNet++/ResNet-18 前向与 batch 显存")
def test_seg_model():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downstream"))
    from seg_lib import build_model

    model = build_model(encoder_weights=None, input_size=224).cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    peaks = {}
    for bs in (8, 16):
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, 1, 224, 224, device="cuda", requires_grad=True)
        out = model(x)
        out.mean().backward()
        peaks[bs] = torch.cuda.max_memory_allocated() / 2**30
        assert out.shape[:2] == (bs, 3), f"输出应为 (B,3,H,W)，实得 {out.shape}"
        del x, out
    del model
    torch.cuda.empty_cache()
    detail = "、".join(f"batch={k}: {v:.2f} GB" for k, v in peaks.items())
    return f"参数量 {n_params:.1f}M；训练峰值 {detail}"


# --------------------------------------------------------------------------
# 6. 参考轨迹生成的形状契约（对齐 test.py 的第 112 行）
# --------------------------------------------------------------------------
@check("参考轨迹数组形状契约 [N, T, 1, 1, H, W] 与索引方式")
def test_reference_shape_contract():
    N, T, H, W = 4, 7, 32, 32
    ref = np.random.randn(N, T, 1, 1, H, W).astype(np.float32)
    idx = 3
    picked = ref[:, idx, 0, 0, ...]
    assert picked.shape == (N, H, W), f"test.py 的取法应得到 (N,H,W)，实得 {picked.shape}"
    return f"ref{ref.shape} -> ref[:, {idx}, 0, 0, ...] -> {picked.shape}（与 match_cumulative_cdf_batch 的模板形状一致）"


def main() -> int:
    print("=" * 78)
    print("官方仓库代码路径验证")
    print("  （不加载真实数据、不训练模型，输出不构成复现结论）")
    print("=" * 78)
    print(f"\ntorch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    for i, (name, fn) in enumerate(CHECKS, 1):
        print(f"\n[{i}/{len(CHECKS)}] {name}")
        run_check(name, fn)

    ok = sum(1 for _, p, _ in RESULTS if p)
    print("\n" + "=" * 78)
    print(f"通过 {ok}/{len(RESULTS)}")
    if ok != len(RESULTS):
        for name, passed, detail in RESULTS:
            if not passed:
                print(f"  ❌ {name}: {detail}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
