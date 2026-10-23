#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DL-EX-025 B1a 对比图：D3_ema（128 从零）vs B1a LoRA（512 预训练底座）

产出两张：
  b1a_vs_d3_same256.png   两者缩放到 256 同尺寸并排（公平比细节密度）
  b1a_vs_d3_native.png    B1a 原 512 / D3 放大到 512（比最终交付观感）
"""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DDPM = Path(__file__).resolve().parents[3] / "deep_learning" / "Fundus-Diffusion" / "ddpm"
D3_DIR = DDPM / "judgment_20261018" / "D3_ema" / "singles"
B1A_DIR = ROOT / "judgment" / "B1a_lora" / "singles"
N = 8
PAD = 6
CAP = 26


def load_grid(d, side, label):
    files = sorted(d.glob("*.png"))
    assert len(files) >= N, f"{d} only {len(files)} images"
    ims = [Image.open(f).convert("RGB").resize((side, side), Image.BICUBIC) for f in files[:N]]
    return ims, label


def sheet(rows, side, out):
    """rows = [(images, label), ...] 横排每行 N 张"""
    W = N * side + (N + 1) * PAD
    H = len(rows) * (side + CAP + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    y = PAD
    for ims, label in rows:
        dr.text((PAD, y), label, fill="black")
        y += CAP
        for i, im in enumerate(ims):
            canvas.paste(im, (PAD + i * (side + PAD), y))
        y += side + PAD
    canvas.save(out)
    print(f"saved {out} {canvas.size}")


d3, l3 = load_grid(D3_DIR, 256, "")
b1, lb = load_grid(B1A_DIR, 256, "")
sheet([(d3, "D3_ema  (from-scratch 128px, EMA, seed42-49, DDIM100) - upscaled to 256"),
       (b1, "B1a LoRA (SD2.1-base 512 + KylianSu LoRA, seed42-49, DDIM100) - downscaled to 256")],
      256, ROOT / "judgment" / "b1a_vs_d3_same256.png")

d3b, _ = load_grid(D3_DIR, 512, "")
b1b, _ = load_grid(B1A_DIR, 512, "")
sheet([(d3b, "D3_ema  (from-scratch 128px) - upscaled to 512"),
       (b1b, "B1a LoRA (native 512)")],
      512, ROOT / "judgment" / "b1a_vs_d3_native.png")
