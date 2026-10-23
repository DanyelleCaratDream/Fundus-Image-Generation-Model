#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1f 生成线对决：D3_ema（±SDEdit t0=200）vs B1b_500（±KylianSu 精修 0.25）vs 真实

分辨率诚实标注：D3 原生 128px（双三次放大到 256 展示）；B1 原生 512px（缩到 256）。
所有行同槽位并排，仅供"专家眼"预览对决；放大到 256 的 D3 并非 SR 交付口径。

用法
    F:/python/python.exe b1f_head_to_head.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DDPM = Path(__file__).resolve().parents[3] / "deep_learning" / "Fundus-Diffusion" / "ddpm"
J = DDPM / "judgment_20261020"  # 跨线对比：D3/CFM 图仍在 diffusion
S3 = DDPM / "sdedit_D3_ema"
ROWS = [
    ((ROOT / "b1b_data/severe512").glob("*.jpg"),
     "real severe (teacher train, 512)"),
    (sorted(S3.glob("orig/sample_*.png")),
     "D3_ema raw (native 128px, bicubic up)"),
    (sorted(S3.glob("t00200/sample_*.png")),
     "D3_ema + SDEdit t0=200 (native 128px, bicubic up)"),
    (sorted((J / "B1b_lora_500/singles").glob("sample_*.png")),
     "B1b step500 raw (native 512px)"),
    (sorted((J / "B1d_refine_s025/singles").glob("sample_*.png")),
     "B1b step500 + KylianSu refine 0.25 (native 512px)"),
]
S, PAD, CAP, N = 256, 6, 24, 8

W = N * S + (N + 1) * PAD
H = len(ROWS) * (S + CAP + PAD) + PAD
canvas = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(canvas)
y = PAD
for files, label in ROWS:
    dr.text((PAD, y), label, fill="black"); y += CAP
    for i, f in enumerate(list(files)[:N]):
        canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC), (PAD + i * (S + PAD), y))
    y += S + PAD
out = J / "b1f_head_to_head.png"
canvas.save(out)
print(f"saved {out} {canvas.size}")
