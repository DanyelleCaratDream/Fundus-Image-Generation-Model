#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DL-EX-025 B1b 推理：SD2.1-base + 我们自训的 severe LoRA

与 b1a_lora_infer.py 的区别：LoRA 是 peft 格式（b1b_lora_train.py 产出），
用 peft 包装 UNet 后 load_state_dict，再用 diffusers 管线出图。

用法
    F:/python/python.exe b1b_lora_infer.py --lora b1b_lora/lora_step_01500.safetensors \
        --num 8 --seed 42 --steps 100 --out judgment_20261020/B1b_lora/singles
"""
import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

BASE = Path(r"D:\hf_cache\hub\models--LanguageMachines--stable-diffusion-2-1-base\snapshots"
            r"\2579f99c90b5a3019fc115ee9566d5ec4f17fccf")
PROMPT_DEFAULT = ("a fundus image of a human retina with severe diabetic retinopathy, "
                  "showing hemorrhages and hard exudates")


def build_pipe(lora_path, rank=32, alpha=32):
    tokenizer = CLIPTokenizer.from_pretrained(str(BASE / "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(str(BASE / "text_encoder"), torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(str(BASE / "vae"), torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(str(BASE / "unet"), torch_dtype=torch.float16)

    cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                     target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet = get_peft_model(unet, cfg)
    sd = load_file(str(lora_path))
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print(f"[lora] loaded {Path(lora_path).name}: {len(sd)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(str(BASE / "scheduler")),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompt", default=PROMPT_DEFAULT)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pipe = build_pipe(a.lora, a.rank, a.alpha)
    for i in range(a.num):
        seed = a.seed + i
        g = torch.Generator(device="cuda").manual_seed(seed)
        img = pipe(prompt=a.prompt, num_inference_steps=a.steps, guidance_scale=a.cfg,
                   width=512, height=512, generator=g).images[0]
        fp = out / f"sample_{i}.png"
        img.save(fp)
        print(f"[ok] {fp.name} seed={seed}")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
