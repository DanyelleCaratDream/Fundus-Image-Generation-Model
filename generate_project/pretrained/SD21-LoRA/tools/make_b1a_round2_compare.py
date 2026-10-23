#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1a 第二轮对比：原始 / 第一轮色彩迁移 / 负向prompt+CFG9 / 第二轮色彩迁移"""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DDPM = Path(__file__).resolve().parents[3] / "deep_learning" / "Fundus-Diffusion" / "ddpm" / "judgment_20261020"
ROWS = [
    (ROOT / "B1a_lora" / "singles", "R1 gen: cfg7.5, no neg prompt (raw)"),
    (ROOT / "B1a_lora_ct" / "singles", "R1 + color transfer (L full, a/b shift only)"),
    (ROOT / "B1a_lora_neg" / "singles", "R2 gen: cfg9.0 + negative prompt (raw)"),
    (ROOT / "B1a_lora_neg_ct" / "singles", "R2 + color transfer"),
]
S, PAD, CAP, N = 256, 6, 24, 8

W = N * S + (N + 1) * PAD
H = len(ROWS) * (S + CAP + PAD) + PAD
canvas = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(canvas)
y = PAD
for d, label in ROWS:
    dr.text((PAD, y), label, fill="black"); y += CAP
    for i, f in enumerate(sorted(d.glob("*.png"))[:N]):
        canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC), (PAD + i * (S + PAD), y))
    y += S + PAD
out = ROOT / "b1a_round2_compare.png"
canvas.save(out)
print(f"saved {out} {canvas.size}")
