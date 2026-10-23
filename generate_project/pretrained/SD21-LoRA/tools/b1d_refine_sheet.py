#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1d 顺序精修对比：真实 / B1b_500 原图 / 精修 0.25 / 精修 0.35（同种子 42-49）"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DDPM = Path(__file__).resolve().parents[3] / "deep_learning" / "Fundus-Diffusion" / "ddpm"
REAL = ROOT / "data" / "b1b_data" / "severe512"
J = ROOT / "judgment"
ROWS = [
    (REAL.glob("*.jpg"), "real severe (teacher train, 512)"),
    ((J / "B1b_lora_500" / "singles").glob("*.png"), "B1b step500 (no refine)"),
    ((J / "B1d_refine_s025" / "singles").glob("*.png"), "refine strength 0.25 (KylianSu img2img)"),
    ((J / "B1d_refine_s035" / "singles").glob("*.png"), "refine strength 0.35 (KylianSu img2img)"),
]
S, PAD, CAP, N = 256, 6, 24, 8

W = N * S + (N + 1) * PAD
H = len(ROWS) * (S + CAP + PAD) + PAD
canvas = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(canvas)
y = PAD
for files, label in ROWS:
    dr.text((PAD, y), label, fill="black"); y += CAP
    for i, f in enumerate(sorted(files)[:N]):
        canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC), (PAD + i * (S + PAD), y))
    y += S + PAD
out = J / "b1d_refine_compare.png"
canvas.save(out)
print(f"saved {out} {canvas.size}")
