#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tiger-Model 零样本观察：用官方预训练权重直接在眼底提示词上生成

目的不是“跑出好图”，而是回答：一个在**超声图 + 文本报告**上训练的文本引导扩散，
迁移到眼底域时还能保留多少结构/语义。预期多为负结果，如实记录。

用法（GPU 空出后）
    python tools/tiger_zeroshot.py --num 4 --steps 30 --out results/tiger_zeroshot
"""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                     # TextGuidedDiffusion/
BASE = ROOT / "models/tiger/model/pretrain"  # diffusers 管线目录

PROMPTS = {
    "severe": "a fundus photograph of a retina with severe diabetic retinopathy, "
              "hemorrhages and hard exudates",
    "normal": "a fundus photograph of a normal retina",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results/tiger_zeroshot"))
    a = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        str(BASE), torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    print("[ok] Tiger pretrain pipeline loaded", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, prompt in PROMPTS.items():
        imgs = []
        for i in range(a.num):
            g = torch.Generator(device="cuda").manual_seed(a.seed + i)
            im = pipe(prompt, num_inference_steps=a.steps, guidance_scale=a.cfg,
                      width=a.size, height=a.size, generator=g).images[0]
            im.save(out / f"{tag}_{i}.png")
            imgs.append(im)
            print(f"[ok] {tag}_{i}", flush=True)
        rows.append((tag, imgs))

    S, PAD, CAP = 256, 6, 24
    canvas = Image.new("RGB", (a.num * S + (a.num + 1) * PAD,
                               len(rows) * (S + CAP + PAD) + PAD), "white")
    dr = ImageDraw.Draw(canvas)
    y = PAD
    for tag, imgs in rows:
        dr.text((PAD, y), f"Tiger zero-shot: {tag}", fill="black"); y += CAP
        for j, im in enumerate(imgs):
            canvas.paste(im.resize((S, S)), (PAD + j * (S + PAD), y))
        y += S + PAD
    canvas.save(out / "tiger_zeroshot_sheet.png")
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
