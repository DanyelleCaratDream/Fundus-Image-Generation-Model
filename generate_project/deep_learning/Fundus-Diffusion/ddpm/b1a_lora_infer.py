#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL-EX-025 B1a 第一步：SD2.1-base + KylianSu LoRA 文本生成图（512 原生，零训练推理）

不含 ControlNet——套件代码在 GitHub（KylianSu/reproduce_dr），本脚本只用标准 diffusers 接口：
    StableDiffusionPipeline.from_pretrained(base)  ->  pipe.load_lora_weights(lora_dir)

用法：
    # 探路：4 个候选 prompt × 2 张
    python b1a_lora_infer.py --mode probe --out results_b1a/probe

    # 正式：单 prompt × 8 张，seed 42 与 D3_ema 判定图对齐
    python b1a_lora_infer.py --mode final --prompt "..." --out judgment_20261020/B1a_lora/singles
"""
import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

BASE_DIR = Path(r"D:\hf_cache\hub\models--LanguageMachines--stable-diffusion-2-1-base\snapshots"
                r"\2579f99c90b5a3019fc115ee9566d5ec4f17fccf")
LORA_DIR = Path(r"D:\hf_cache\hub\models--KylianSu--vessel-bezier-retinal-weights\snapshots"
                r"\3875b7e4eb8d146c58525c77b61e67a72fa147c7\weights\sd21_lora_fundus_v2")

PROBE_PROMPTS = [
    "a color fundus photograph of the retina with diabetic retinopathy",
    "a fundus image of a human retina, showing blood vessels and optic disc",
    "retinal fundus photography, color, clinical, high detail",
    "a photograph of the retina",
]


def build_pipe(lora_scale=1.0):
    pipe = StableDiffusionPipeline.from_pretrained(
        str(BASE_DIR), torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(LORA_DIR))
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, lora_scale


def gen(pipe, prompt, seed, steps, lora_scale, out_path, negative_prompt=None, cfg=7.5):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = {}
    if negative_prompt:
        kw["negative_prompt"] = negative_prompt
    img = pipe(
        prompt=prompt, num_inference_steps=steps, guidance_scale=cfg,
        width=512, height=512, generator=g,
        cross_attention_kwargs={"scale": lora_scale} if lora_scale != 1.0 else None,
        **kw,
    ).images[0]
    img.save(out_path)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "final"], default="final")
    ap.add_argument("--prompt", default=PROBE_PROMPTS[0])
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--negative_prompt", default=None)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] base={BASE_DIR}")
    print(f"[load] lora={LORA_DIR}")
    pipe, ls = build_pipe(args.lora_scale)
    print(f"[gpu] {torch.cuda.get_device_name(0)}  "
          f"alloc={torch.cuda.memory_allocated()/2**30:.2f}GB")

    if args.mode == "probe":
        # 每个 prompt 出 2 张（seed 42/43），文件名带 prompt 序号
        for pi, p in enumerate(PROBE_PROMPTS):
            for j, s in enumerate([42, 43]):
                fp = out / f"p{pi}_s{s}.png"
                gen(pipe, p, s, args.steps, ls, fp)
                print(f"[ok] {fp.name}  <- {p}")
    else:
        for i in range(args.num):
            seed = args.seed + i
            fp = out / f"sample_{i}.png"
            gen(pipe, args.prompt, seed, args.steps, ls, fp,
                negative_prompt=args.negative_prompt, cfg=args.cfg)
            print(f"[ok] {fp.name}  seed={seed}  neg={bool(args.negative_prompt)} cfg={args.cfg}")

    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
