#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TTA-Flow 思路的移植：轨迹对齐直方图匹配（推理期，无需重训）

来源：Hucke et al., "Test-Time Adaptation in OCT Using Trajectory-Aligned
Time-Independent Flow"（代码 code/external/tta-flow）。

流程
1. 用当前 CFM 模型生成 N 条参考轨迹，记录第 s_target 步的样本；
2. 对这 N 个样本做逐通道平均直方图（CDF）；
3. 对每张测试图：把直方图匹配到参考 CDF，然后从 s_target 启动 Euler 积分到 t=1；
4. 输出 before/after 对照。

用途（我们）：①把老师真实 severe 图"投影"到 CFM 流形，做域适应/可视化；
②对生成图做非参数色彩校准（与 Reinhard 对照）。不改模型、不加训练。

用法
    python tools/tta_hist_match.py --ckpt models/cfm256_final_model.pth \
        --src <测试图目录> --num_ref 16 --s_target 0.5 --steps 100 --out judgment/tta_proj
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

import sys
ROOT = Path(__file__).resolve().parents[1]          # CFM/
DL = ROOT.parents[1]                                 # deep_learning/
sys.path.insert(0, str(DL / "Fundus-Diffusion" / "ddpm"))

from train import UNet, GaussianDiffusion, check_environment  # noqa: E402
from generate import load_condition_images  # noqa: E402


def load_model(ckpt_path, base_dim, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ckpt.get("args", None)
    use_ssn = getattr(ck_args, "use_scale_shift_norm", True) if ck_args is not None else True
    model = UNet(T=1000, channels=3, base_dim=base_dim, dim_mults=(1, 2, 3, 4),
                 attn_layers=(2,), dropout=0.0, cond_channels=1,
                 use_scale_shift_norm=use_ssn)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if "ema_state_dict" in ckpt and ckpt.get("has_ema", False):
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    return model.to(device).eval(), ck_args


@torch.no_grad()
def euler_until(model, diffusion, x, t_start, steps, cond, time_align=False):
    """从 t_start 积分到 1，返回最终样本"""
    b = x.shape[0]
    dt = (1.0 - t_start) / steps
    for i in range(steps):
        t_cont = torch.full((b,), t_start + i * dt, device=x.device)
        t_idx = diffusion.t_cont_to_index(t_cont, align=time_align)
        v = model(x, t_idx, cond=cond) if cond is not None else model(x, t_idx)
        x = x + v * dt
    return x.clamp(-1.0, 1.0)


def avg_cdf(samples, bins=256):
    """samples: [N,3,H,W] in [-1,1] → 每通道平均 CDF（256 bins）"""
    arr = ((samples.clamp(-1, 1) + 1) / 2 * (bins - 1)).round().to(torch.int64).cpu().numpy()
    cdfs = []
    for c in range(arr.shape[1]):
        hist = np.zeros(bins, dtype=np.float64)
        for n in range(arr.shape[0]):
            hist += np.bincount(arr[n, c].ravel(), minlength=bins)
        cdf = np.cumsum(hist) / hist.sum()
        cdfs.append(cdf)
    return cdfs


def match_to_cdf(img, cdf, bins=256):
    """img: [3,H,W] tensor in [-1,1]；逐通道 CDF 匹配，返回同形状 tensor"""
    arr = ((img.clamp(-1, 1) + 1) / 2 * (bins - 1)).round().to(torch.int64).cpu().numpy()
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        vals, inv, counts = np.unique(arr[c].ravel(), return_inverse=True, return_counts=True)
        q = np.cumsum(counts) / counts.sum()
        mapped = np.interp(q, cdf[c], np.linspace(0, bins - 1, bins))
        out[c] = mapped[inv].reshape(arr[c].shape)
    return torch.from_numpy(out / (bins - 1) * 2 - 1).to(img.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True, help="测试图目录（png/jpg）")
    ap.add_argument("--cond", default=str(ROOT / "data/conditions_combined_256"))
    ap.add_argument("--num_ref", type=int, default=16)
    ap.add_argument("--s_target", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--base_dim", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if check_environment(a.no_cuda) else "cpu")
    model, ck_args = load_model(a.ckpt, a.base_dim, device)
    time_align = bool(getattr(ck_args, "cfm_time_align", False))
    diffusion = GaussianDiffusion(timesteps=1000).to(device)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # 1) 参考轨迹：从噪声积分，记录 s_target 步样本
    torch.manual_seed(a.seed)
    shape = (a.num_ref, 3, a.img_size, a.img_size)
    conds_ref = load_condition_images(a.cond, a.img_size, a.num_ref).to(device)
    x = torch.randn(shape, device=device)
    dt = a.s_target / max(1, int(a.s_target * a.steps))
    n0 = max(1, int(a.s_target * a.steps))
    for i in range(n0):
        t_cont = torch.full((a.num_ref,), i * dt, device=device)
        t_idx = diffusion.t_cont_to_index(t_cont, align=time_align)
        x = x + model(x, t_idx, cond=conds_ref) * dt
    refs = x.clamp(-1, 1)
    cdfs = avg_cdf(refs)
    save_image(torch.cat([refs[:4]]), out / "reference_samples.png", nrow=4, normalize=True)
    print(f"[ref] {a.num_ref} trajectories @ s={a.s_target}", flush=True)

    # 2) 测试图 → 直方图匹配 → 从 s_target 重建
    files = sorted([p for p in Path(a.src).iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    assert files, f"no images in {a.src}"
    conds = load_condition_images(a.cond, a.img_size, len(files)).to(device)
    before, after = [], []
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB").resize((a.img_size, a.img_size), Image.LANCZOS)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        t = t.to(device)
        before.append(t)
        t_m = match_to_cdf(t, cdfs)
        y = euler_until(model, diffusion, t_m[None], a.s_target, a.steps,
                        conds[i:i + 1], time_align=time_align)[0]
        after.append(y)
        save_image(y, out / f.name, normalize=True)
        print(f"[ok] {f.name}", flush=True)

    # 3) 对照表
    S, PAD, CAP, N = 224, 6, 24, len(files)
    canvas = Image.new("RGB", (N * S + (N + 1) * PAD, 2 * (S + CAP + PAD) + PAD), "white")
    dr = ImageDraw.Draw(canvas)
    y = PAD
    for label, imgs in [("input (test domain)", before), ("after trajectory-aligned HM + ODE", after)]:
        dr.text((PAD, y), label, fill="black"); y += CAP
        for j, t in enumerate(imgs):
            arr = ((t.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            canvas.paste(Image.fromarray(arr).resize((S, S)), (PAD + j * (S + PAD), y))
        y += S + PAD
    canvas.save(out / "tta_compare.png")
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
