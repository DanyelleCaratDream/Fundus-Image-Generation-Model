#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DL-EX-025 B1c：KylianSu 细节 LoRA + 自训 severe LoRA 加权合并推理

两份 LoRA rank 不同（KylianSu r64 diffusers 格式、自训 r32 peft 格式），
不叠 adapter，直接把各自 ΔW 算出来加权加回底座（两份 scale 均为 1）：
    W_merged = W_base + w_k · (up @ down) + w_o · (B @ A)
同种子 42-49 扫配比，与 B1b_500 / 真实 severe 并排出对比图。

用法
    F:/python/python.exe b1c_merge_infer.py                          # 默认三组配比
    F:/python/python.exe b1c_merge_infer.py --combos "1.0,1.0"       # 单组
"""
import argparse
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

BASE = Path(r"D:\hf_cache\hub\models--LanguageMachines--stable-diffusion-2-1-base\snapshots"
            r"\2579f99c90b5a3019fc115ee9566d5ec4f17fccf")
KYLIAN = Path(r"D:\hf_cache\hub\models--KylianSu--vessel-bezier-retinal-weights\snapshots"
              r"\3875b7e4eb8d146c58525c77b61e67a72fa147c7\weights\sd21_lora_fundus_v2"
              r"\pytorch_lora_weights.safetensors")
OURS = Path(__file__).resolve().parents[1] / "results" / "b1b_lora" / "lora_step_00500.safetensors"
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
        time.sleep(60)
        t = gpu_temp()


def deltas_from_kylian(path):
    """diffusers 格式：{mod}.lora.down.weight / {mod}.lora.up.weight，前缀 unet."""
    sd = load_file(str(path))
    raw = {}
    for k, v in sd.items():
        if ".lora.down." in k:
            mod = k.split(".lora.down.")[0].replace("unet.", "", 1)
            raw.setdefault(mod, {})["down"] = v.float()
        elif ".lora.up." in k:
            mod = k.split(".lora.up.")[0].replace("unet.", "", 1)
            raw.setdefault(mod, {})["up"] = v.float()
    out = {m: d["up"] @ d["down"] for m, d in raw.items()}
    return out


def deltas_from_ours(path):
    """peft 格式：{mod}.lora_A.default.weight / {mod}.lora_B.default.weight"""
    sd = load_file(str(path))
    raw = {}
    for k, v in sd.items():
        if ".lora_A." in k:
            mod = k.split(".lora_A.")[0].replace("base_model.model.", "", 1)
            raw.setdefault(mod, {})["A"] = v.float()
        elif ".lora_B." in k:
            mod = k.split(".lora_B.")[0].replace("base_model.model.", "", 1)
            raw.setdefault(mod, {})["B"] = v.float()
    out = {m: d["B"] @ d["A"] for m, d in raw.items()}
    return out


def merge_unet(dk, do, w_k, w_o):
    unet = UNet2DConditionModel.from_pretrained(str(BASE / "unet"), torch_dtype=torch.float32)
    sd = unet.state_dict()
    n_k = n_o = 0
    for mod, delta in dk.items():
        key = f"{mod}.weight"
        assert key in sd, f"kylian delta key not in unet: {key}"
        sd[key] += w_k * delta
        n_k += 1
    for mod, delta in do.items():
        key = f"{mod}.weight"
        assert key in sd, f"ours delta key not in unet: {key}"
        sd[key] += w_o * delta
        n_o += 1
    unet.load_state_dict(sd)
    print(f"[merge] w_k={w_k} applied {n_k} mods, w_o={w_o} applied {n_o} mods "
          f"(kylian={len(dk)}, ours={len(do)})", flush=True)
    return unet.to(torch.float16)


def build_pipe(unet):
    tokenizer = CLIPTokenizer.from_pretrained(str(BASE / "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(str(BASE / "text_encoder"), torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(str(BASE / "vae"), torch_dtype=torch.float16)
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
    ap.add_argument("--combos", nargs="+", default=["1.0,1.0", "1.0,0.6", "0.6,1.0"])
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.5)
    a = ap.parse_args()

    print("[load] computing LoRA deltas", flush=True)
    dk = deltas_from_kylian(KYLIAN)
    do = deltas_from_ours(OURS)
    print(f"[deltas] kylian {len(dk)} mods (rank from up.shape[1] sample: "
          f"{next(iter(dk.values())).shape[1]}), ours {len(do)} mods (rank "
          f"{next(iter(do.values())).shape[1]})", flush=True)
    both = len(set(dk) & set(do))
    print(f"[deltas] modules in both: {both} (kylian-only {len(set(dk)-set(do))}, "
          f"ours-only {len(set(do)-set(dk))})", flush=True)

    for combo in a.combos:
        w_k, w_o = (float(x) for x in combo.split(","))
        tag = f"k{int(w_k*10):02d}_o{int(w_o*10):02d}"
        out = OUT_ROOT / f"B1c_merge_{tag}" / "singles"
        out.mkdir(parents=True, exist_ok=True)
        if all((out / f"sample_{i}.png").exists() for i in range(a.num)):
            print(f"[skip] {tag} already done", flush=True)
            continue

        unet = merge_unet(dk, do, w_k, w_o)
        pipe = build_pipe(unet)
        print(f"[gen] combo {w_k}/{w_o} -> {out}", flush=True)
        for i in range(a.num):
            g = torch.Generator(device="cuda").manual_seed(a.seed + i)
            img = pipe(prompt=PROMPT, num_inference_steps=a.steps, guidance_scale=a.cfg,
                       width=512, height=512, generator=g).images[0]
            img.save(out / f"sample_{i}.png")
            print(f"[ok] {tag} sample_{i} seed={a.seed + i} temp={gpu_temp()}C", flush=True)
            cool_wait()
        del pipe, unet
        torch.cuda.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
