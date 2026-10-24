#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索条件 CFM 的采样：每张图用一个 severe anchor 的重心条件向量

用法
    python tools/sample_feat.py --ckpt results/cfm128_featC/models/final_model.pth \
        --conds data/handcrafted/conds_severe.npy --num 8 --img_size 128 --out judgment/featC
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

CFM = Path(__file__).resolve().parents[1]
DL = CFM.parents[1]
sys.path.insert(0, str(DL / "Fundus-Diffusion" / "ddpm"))
from train import UNet, GaussianDiffusion  # noqa: E402
from generate import load_condition_images  # noqa: E402


class FeatWrap:
    def __init__(self, model, vec):
        self.model, self.vec = model, vec

    def __call__(self, x, t, cond=None):
        v = self.vec[None].expand(x.shape[0], -1).to(x.device)
        return self.model(x, t, cond=cond, text_emb=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--base_dim", type=int, default=128)
    ap.add_argument("--cond_path", default=str(CFM / "data/conditions_combined_128"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    ck_args = ck.get("args", None)
    feat_dir = getattr(ck_args, "feat_cond_dir", None)
    dim = int(np.load(Path(feat_dir) / "feats.npy", mmap_mode="r").shape[1]) if feat_dir else 15
    use_ssn = getattr(ck_args, "use_scale_shift_norm", True) if ck_args else True
    time_align = bool(getattr(ck_args, "cfm_time_align", False)) if ck_args else False

    model = UNet(T=1000, channels=3, base_dim=a.base_dim, dim_mults=(1, 2, 3, 4),
                 attn_layers=(2,), dropout=0.0, cond_channels=1,
                 use_scale_shift_norm=use_ssn, text_dim=dim)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    if "ema_state_dict" in ck and ck.get("has_ema", False):
        model.load_state_dict(ck["ema_state_dict"], strict=False)
    model.to(device).eval()
    print(f"[ok] loaded {a.ckpt} (feat_dim={dim})")

    conds = torch.from_numpy(np.load(a.conds)).float().to(device)
    cond_imgs = load_condition_images(a.cond_path, a.img_size, a.num).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    imgs = []
    with torch.no_grad():
        for i in range(a.num):
            torch.manual_seed(a.seed + i)
            wrap = FeatWrap(model, conds[i % len(conds)])
            y = diffusion.euler_sample(wrap, (1, 3, a.img_size, a.img_size), device,
                                       sampling_steps=a.steps, cond=cond_imgs[i:i + 1],
                                       time_align=time_align)
            save_image(y[0], out / f"sample_{i}.png", normalize=True)
            arr = ((y[0].clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            imgs.append(Image.fromarray(arr))
            print(f"[ok] {i}", flush=True)

    S, PAD = 256, 6
    canvas = Image.new("RGB", (a.num * S + (a.num + 1) * PAD, S + 2 * PAD), "white")
    for i, im in enumerate(imgs):
        canvas.paste(im.resize((S, S)), (PAD + i * (S + PAD), PAD))
    canvas.save(out / "grid.png")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
