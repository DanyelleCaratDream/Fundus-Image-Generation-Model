#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1c 合并对比图：真实 severe / B1b_500 / 三组 LoRA 合并（同种子 42-49）"""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
REAL = ROOT / "b1b_data" / "severe512"
J = ROOT / "judgment_20261020"
ROWS = [
    (REAL.glob("*.jpg"), "real severe (teacher train, 512)"),
    ((J / "B1b_lora_500" / "singles").glob("*.png"), "B1b ours r32 step500 (no merge)"),
    ((J / "B1c_merge_k10_o10" / "singles").glob("*.png"), "merge K1.0 + O1.0"),
    ((J / "B1c_merge_k10_o06" / "singles").glob("*.png"), "merge K1.0 + O0.6"),
    ((J / "B1c_merge_k06_o10" / "singles").glob("*.png"), "merge K0.6 + O1.0"),
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
out = J / "b1c_merge_compare.png"
canvas.save(out)
print(f"saved {out} {canvas.size}")
