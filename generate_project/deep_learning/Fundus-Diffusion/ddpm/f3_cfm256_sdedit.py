#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F3 原生 256 的 CFM 自精修（不做 SR——原生 256 已是交付分辨率）

底图 = judgment_20261020/D5_cfm256_ema/singles（作者给出 97/100 的那批）
加噪 = 线性路径 x_t = t_start·x1 + (1-t_start)·noise；Euler t_start→1
档位：0.8（≈DDPM t0=200）、0.7（≈300）

用法
    F:/python/python.exe f3_cfm256_sdedit.py
"""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

from train import UNet, GaussianDiffusion, check_environment
from generate import load_condition_images

ROOT = Path(__file__).resolve().parent
J = ROOT / "judgment_20261020"
BASE_DIR = J / "D5_cfm256_ema" / "singles"
OUT_ROOT = J / "F3_cfm256_sdedit"


@torch.no_grad()
def refine(model, diffusion, x1, t_start, steps, cond, align=False):
    b = x1.shape[0]
    noise = torch.randn_like(x1)
    x = t_start * x1 + (1.0 - t_start) * noise
    dt = (1.0 - t_start) / steps
    for i in range(steps):
        t_cont = torch.full((b,), t_start + i * dt, device=x1.device)
        t_idx = diffusion.t_cont_to_index(t_cont, align=align)
        v = model(x, t_idx, cond=cond) if cond is not None else model(x, t_idx)
        x = x + v * dt
    return x.clamp(-1.0, 1.0)


def load_bases(n, base_dir=None):
    files = sorted((Path(base_dir) if base_dir else BASE_DIR).glob("sample_*.png"))[:n]
    assert files, f"no base images in {BASE_DIR}"
    out = []
    for f in files:
        a = torch.from_numpy(
            __import__("numpy").asarray(Image.open(f).convert("RGB"), dtype="float32") / 255.0
        ).permute(2, 0, 1)
        out.append(a * 2.0 - 1.0)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT.parent.parent / "Fundus-FlowMatching/CFM/models/cfm256_final_model.pth"))
    ap.add_argument("--cond_path", default=str(ROOT / "conditions_combined_256"))
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--base_dim", type=int, default=128)
    ap.add_argument("--t_start", type=float, nargs="+", default=[0.8, 0.7])
    ap.add_argument("--denoise_steps", type=int, default=100)
    ap.add_argument("--noise_seed", type=int, default=1234)
    ap.add_argument("--base_dir", default=str(BASE_DIR))
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--no_cuda", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if check_environment(a.no_cuda) else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    ck_args = ckpt.get("args", None)
    use_ssn = getattr(ck_args, "use_scale_shift_norm", True) if ck_args is not None else True
    model = UNet(T=1000, channels=3, base_dim=a.base_dim, dim_mults=(1, 2, 3, 4),
                 attn_layers=(2,), dropout=0.0, cond_channels=1,
                 use_scale_shift_norm=use_ssn,
        use_time_cond=not bool(getattr(ck_args, "no_time_cond", False)))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if "ema_state_dict" in ckpt and ckpt.get("has_ema", False):
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        print("[OK] EMA weights", flush=True)
    model.to(device).eval()

    conds = load_condition_images(a.cond_path, a.img_size, a.num_images).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    x1 = load_bases(a.num_images, a.base_dir).to(device)

    outputs = {}
    for ts in a.t_start:
        torch.manual_seed(a.noise_seed)
        print(f"[SDEDIT] t_start={ts} {a.denoise_steps} Euler steps", flush=True)
        outputs[ts] = refine(model, diffusion, x1, ts, a.denoise_steps, conds,
                             align=bool(getattr(ck_args, "cfm_time_align", False)))
        d = Path(a.out_root) / f"t{int(ts*1000):04d}" / "singles"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.num_images):
            save_image(outputs[ts][i], d / f"sample_{i}.png", normalize=True)
        print(f"[ok] {d}", flush=True)

    rows = [(sorted((ROOT / "b1b_data/severe512").glob("*.jpg"))[:8], "real severe (512)"),
            (sorted(Path(a.base_dir).glob("*.png"))[:8], "CFM@256 base")]
    for ts in a.t_start:
        rows.append((sorted((Path(a.out_root) / f"t{int(ts*1000):04d}/singles").glob("*.png"))[:8],
                     f"CFM@256 + SDEdit({ts})"))
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
    out = J / "f3_cfm256_sdedit_compare.png"
    canvas.save(out)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
