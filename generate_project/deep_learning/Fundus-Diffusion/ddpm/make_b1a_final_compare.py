#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1a 最终三方对比：D3_ema / B1a 原始 / B1a+色彩迁移（同 256 尺寸，各 8 张）"""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ROWS = [
    (ROOT / "judgment_20261018" / "D3_ema" / "singles",
     "D3_ema  (from-scratch 128px, EMA) - upsample 256"),
    (ROOT / "judgment_20261020" / "B1a_lora" / "singles",
     "B1a LoRA (SD2.1-base 512, native, no color fix) - downsample 256"),
    (ROOT / "judgment_20261020" / "B1a_lora_ct" / "singles",
     "B1a LoRA + Reinhard LAB transfer (anchor = real B-domain train stats) - 256"),
]
S, PAD, CAP, N = 256, 6, 24, 8

W = N * S + (N + 1) * PAD
H = len(ROWS) * (S + CAP + PAD) + PAD
canvas = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(canvas)
y = PAD
for d, label in ROWS:
    dr.text((PAD, y), label, fill="black"); y += CAP
    files = sorted(d.glob("*.png"))[:N]
    for i, f in enumerate(files):
        canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC), (PAD + i * (S + PAD), y))
    y += S + PAD
out = ROOT / "judgment_20261020" / "b1a_final_three_way.png"
canvas.save(out)
print(f"saved {out} {canvas.size}")
