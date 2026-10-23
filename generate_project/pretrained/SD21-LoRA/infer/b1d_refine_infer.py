#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DL-EX-025 B1d 顺序精修（路线 B）：B1b_500 出图 → img2img 挂 KylianSu LoRA 低强度补高频

与 B1c 加权合并的本质区别：内容先锁进 latent，KylianSu 只参与剩余去噪步。
强度 s 意味着 (1-s) 比例的步数保持原 latent——结构（病灶/构图/视盘）保留，只补高频细节。

用法
    F:/python/python.exe b1d_refine_infer.py                  # 默认 0.25/0.35 两档 × 8 张
    F:/python/python.exe b1d_refine_infer.py --strengths 0.3
"""
import argparse
import subprocess
from pathlib import Path

import torch
from diffusers import (AutoencoderKL, DDIMScheduler, StableDiffusionImg2ImgPipeline,
                       UNet2DConditionModel)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

BASE = Path(r"D:\hf_cache\hub\models--LanguageMachines--stable-diffusion-2-1-base\snapshots"
            r"\2579f99c90b5a3019fc115ee9566d5ec4f17fccf")
KYLIAN_DIR = Path(r"D:\hf_cache\hub\models--KylianSu--vessel-bezier-retinal-weights\snapshots"
                  r"\3875b7e4eb8d146c58525c77b61e67a72fa147c7\weights\sd21_lora_fundus_v2")
SRC = Path(__file__).resolve().parent / "judgment_20261020" / "B1b_lora_500" / "singles"
OUT_ROOT = Path(__file__).resolve().parents[1]
DDPM = Path(__file__).resolve().parents[3] / "deep_learning" / "Fundus-Diffusion" / "ddpm" / "judgment_20261020"
PROMPT = ("a fundus image of a human retina with severe diabetic retinopathy, "
          "showing hemorrhages and hard exudates")


def gpu_temp():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            timeout=5, text=True)
        return int(o.strip().splitlines()[0])
    except Exception:
        return -1


def cool_wait(limit=70):
    t = gpu_temp()
    while 0 <= t > limit:
        print(f"[cool] {t}C > {limit}C, sleeping 60s", flush=True)
        import time
        time.sleep(60)
        t = gpu_temp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.35])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src", default=str(SRC), help="输入图目录（默认 B1b_500 的 8 张）")
    ap.add_argument("--out", default=None, help="输出根目录（默认 OUT_ROOT，按强度建子目录）")
    a = ap.parse_args()
    src_dir = Path(a.src)
    out_base = Path(a.out) if a.out else OUT_ROOT

    print("[load] components", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(str(BASE / "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(str(BASE / "text_encoder"), torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(str(BASE / "vae"), torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(str(BASE / "unet"), torch_dtype=torch.float16)
    pipe = StableDiffusionImg2ImgPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(str(BASE / "scheduler")),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipe.load_lora_weights(str(KYLIAN_DIR))
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    srcs = sorted(src_dir.glob("sample_*.png"))
    assert srcs, f"no source images in {src_dir}"
    for s in a.strengths:
        out = (out_base / f"B1d_refine_s{int(s*100):03d}" / "singles") if a.out is None \
            else (out_base / f"s{int(s*100):03d}" / "singles")
        out.mkdir(parents=True, exist_ok=True)
        print(f"[gen] strength={s} -> {out}", flush=True)
        for i, f in enumerate(srcs):
            g = torch.Generator(device="cuda").manual_seed(a.seed + i)
            img = pipe(prompt=PROMPT, image=Image.open(f).convert("RGB"),
                       strength=s, guidance_scale=a.cfg, num_inference_steps=a.steps,
                       generator=g).images[0]
            img.save(out / f.name)
            print(f"[ok] s={s} {f.name} temp={gpu_temp()}C", flush=True)
            cool_wait()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
