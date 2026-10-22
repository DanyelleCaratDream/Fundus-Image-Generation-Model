#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F2：CFM 自精修（低强度 img2img）+ Real-ESRGAN 交付，一步出对照

CFM 版 SDEdit（与 DDPM 的 sdedit.py 对应，但几何不同）：
    x1 = 底图（judgment_20261020/D3_cfm_ema/singles，作者已过目的 8 张）
    x_t = t_start·x1 + (1-t_start)·noise      # 线性路径上"加噪"
    Euler 积分 t: t_start → 1                  # 重掷细节、保留布局
等价档位：DDPM t0=200 ↔ CFM t_start=0.8；t0=300 ↔ 0.7。

产出：
    judgment_20261020/F2_cfm_sdedit/t0800/singles/   (t_start=0.8)
    judgment_20261020/F2_cfm_sdedit/t0700/singles/   (t_start=0.7)
    judgment_20261020/F2_cfm_sr/{base_256,t0800_256,t0700_256}/
    judgment_20261020/f2_cfm_sdedit_sr_compare.png
"""
import argparse
import subprocess
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

from train import UNet, GaussianDiffusion, check_environment
from generate import load_condition_images
from f1_sr_deliver import RRDBNet, upscale, WEIGHTS

ROOT = Path(__file__).resolve().parent
J = ROOT / "judgment_20261020"
BASE_DIR = J / "D3_cfm_ema" / "singles"
OUT_ROOT = J / "F2_cfm_sdedit"
SR_ROOT = J / "F2_cfm_sr"


def gpu_temp():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            timeout=5, text=True)
        return int(o.strip().splitlines()[0])
    except Exception:
        return -1


@torch.no_grad()
def cfm_refine(model, diffusion, x1, t_start, steps, cond):
    """从线性路径 t_start 处积分到 t=1（Euler）"""
    b = x1.shape[0]
    noise = torch.randn_like(x1)  # 调用方已设 seed
    x = t_start * x1 + (1.0 - t_start) * noise
    dt = (1.0 - t_start) / steps
    for i in range(steps):
        t_cont = torch.full((b,), t_start + i * dt, device=x1.device)
        t_idx = diffusion.t_cont_to_index(t_cont)
        v = model(x, t_idx, cond=cond) if cond is not None else model(x, t_idx)
        x = x + v * dt
        if i % 50 == 0:
            print(f"    step {i}/{steps} temp={gpu_temp()}C", flush=True)
    return x.clamp(-1.0, 1.0)


def load_base_images(n):
    files = sorted(BASE_DIR.glob("sample_*.png"))[:n]
    assert files, f"no base images in {BASE_DIR}"
    imgs = []
    for f in files:
        a = torch.from_numpy(
            __import__("numpy").asarray(Image.open(f).convert("RGB"), dtype="float32") / 255.0
        ).permute(2, 0, 1)
        imgs.append(a * 2.0 - 1.0)
    return torch.stack(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "results_cfm_l1lpips/models/final_model.pth"))
    ap.add_argument("--cond_path", default=str(ROOT / "conditions_combined_128"))
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--base_dim", type=int, default=128)
    ap.add_argument("--t_start", type=float, nargs="+", default=[0.8, 0.7, 0.5])
    ap.add_argument("--denoise_steps", type=int, default=100)
    ap.add_argument("--noise_seed", type=int, default=1234)
    ap.add_argument("--no_cuda", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if check_environment(a.no_cuda) else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    ck_args = ckpt.get("args", None)
    use_ssn = getattr(ck_args, "use_scale_shift_norm", True) if ck_args is not None else True
    model = UNet(T=1000, channels=3, base_dim=a.base_dim, dim_mults=(1, 2, 3, 4),
                 attn_layers=(2,), dropout=0.0, cond_channels=1,
                 use_scale_shift_norm=use_ssn)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if "ema_state_dict" in ckpt and ckpt.get("has_ema", False):
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        print("[OK] EMA 权重", flush=True)
    model.to(device).eval()

    conds = load_condition_images(a.cond_path, a.img_size, a.num_images).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)

    x1 = load_base_images(a.num_images).to(device)
    print(f"[base] {x1.shape} from {BASE_DIR}", flush=True)

    outputs = {}
    for ts in a.t_start:
        torch.manual_seed(a.noise_seed)
        print(f"[SDEDIT] t_start={ts} ({a.denoise_steps} Euler steps)", flush=True)
        outputs[ts] = cfm_refine(model, diffusion, x1, ts, a.denoise_steps, conds)
        d = OUT_ROOT / f"t{int(ts*1000):04d}" / "singles"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.num_images):
            save_image(outputs[ts][i], d / f"sample_{i}.png", normalize=True)
        print(f"[ok] -> {d}", flush=True)

    # ---- SR 交付（base 与各档位都做，256）----
    sr = RRDBNet().cuda().eval()
    sd = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    sr.load_state_dict(sd, strict=False)
    print("[sr] weights loaded", flush=True)

    sr_groups = {"base": x1}
    for ts, t in outputs.items():
        sr_groups[f"t{int(ts*1000):04d}"] = t
    for tag, t in sr_groups.items():
        outdir = SR_ROOT / f"{tag}_256"
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(a.num_images):
            arr = ((t[i].clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
            im = Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())
            upscale(sr, im).resize((256, 256), Image.LANCZOS).save(outdir / f"sample_{i}.png")
        print(f"[sr] {tag} -> {outdir}", flush=True)
    del sr
    torch.cuda.empty_cache()

    # ---- 对照表 ----
    rows = [
        (sorted((ROOT / "b1b_data/severe512").glob("*.jpg"))[:8], "real severe (512)"),
        (sorted((J / "D3_sr/t00200_256").glob("*.png"))[:8], "D3 + SDEdit t0=200 + SR (baseline)"),
        (sorted((J / "D3_cfm_ema/singles").glob("*.png"))[:8], "CFM base 128"),
        (sorted((SR_ROOT / "base_256").glob("*.png"))[:8], "CFM base + SR 256"),
        (sorted((SR_ROOT / "t0800_256").glob("*.png"))[:8], "CFM + SDEdit(0.8) + SR 256"),
        (sorted((SR_ROOT / "t0700_256").glob("*.png"))[:8], "CFM + SDEdit(0.7) + SR 256"),
        (sorted((SR_ROOT / "t0500_256").glob("*.png"))[:8], "CFM + SDEdit(0.5) + SR 256"),
    ]
    S, PAD, CAP, N = 256, 6, 24, 8
    canvas = Image.new("RGB", (N * S + (N + 1) * PAD, len(rows) * (S + CAP + PAD) + PAD), "white")
    dr = ImageDraw.Draw(canvas)
    y = PAD
    for files, label in rows:
        dr.text((PAD, y), label, fill="black"); y += CAP
        for i, f in enumerate(files[:N]):
            canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC),
                         (PAD + i * (S + PAD), y))
        y += S + PAD
    out = J / "f2_cfm_sdedit_sr_compare.png"
    canvas.save(out)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
