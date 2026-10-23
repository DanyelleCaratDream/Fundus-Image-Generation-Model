#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1b2 检查点扫描对比：真实 / 旧 B1b_500（整图训练）/ 新 patch 训练 5 档检查点（同种子 42-49）"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DDPM = Path(__file__).resolve().parents[3] / "deep_learning" / "Fundus-Diffusion" / "ddpm"
REAL = ROOT / "data" / "b1b_data" / "severe512"
J = ROOT / "judgment"
ROWS = [
    (REAL.glob("*.jpg"), "real severe (teacher train, 512)"),
    ((J / "B1b_lora_500" / "singles").glob("*.png"), "old B1b whole-img step500"),
    ((J / "B1b2_ckpt_00500" / "singles").glob("*.png"), "patch-train step500"),
    ((J / "B1b2_ckpt_00750" / "singles").glob("*.png"), "patch-train step750"),
    ((J / "B1b2_ckpt_01000" / "singles").glob("*.png"), "patch-train step1000"),
    ((J / "B1b2_ckpt_01250" / "singles").glob("*.png"), "patch-train step1250"),
    ((J / "B1b2_ckpt_01500" / "singles").glob("*.png"), "patch-train step1500"),
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
out = J / "b1b2_ckpt_compare.png"
canvas.save(out)
print(f"saved {out} {canvas.size}")
