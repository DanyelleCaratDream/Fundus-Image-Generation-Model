#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DL-EX-025 B1b：老师 severe 72 张 few-shot LoRA 微调 SD2.1-base

设计要点
- 底座冻结，只训 UNet 的 LoRA（peft），rank/alpha 可调（默认 32/32）
- noise_offset=0.1（KylianSu 配方），fp16 混合精度
- 温度守护：每 --temp_every 步读 GPU 温度，超过 --temp_limit 就休眠到降下来（≤70°C 纪律）
- 每 --save_every 步存一份 LoRA safetensors（可续训）
- 固定 prompt：72 张全是 severe，单一类别条件

用法
    F:/python/python.exe b1b_lora_train.py --steps 1500 --batch 2
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

BASE = Path(r"D:\hf_cache\hub\models--LanguageMachines--stable-diffusion-2-1-base\snapshots"
            r"\2579f99c90b5a3019fc115ee9566d5ec4f17fccf")
DATA = Path(__file__).resolve().parents[1] / "data" / "b1b_data" / "severe512"
OUT = Path(__file__).resolve().parents[1] / "results" / "b1b_lora"
PROMPT = ("a fundus image of a human retina with severe diabetic retinopathy, "
          "showing hemorrhages and hard exudates")


def gpu_temp():
    try:
        import subprocess
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            timeout=5, text=True)
        return int(o.strip().splitlines()[0])
    except Exception:
        return -1


def cool_wait(limit, verbose=True):
    """温度超过 limit 就睡，直到降下来"""
    t = gpu_temp()
    while 0 <= t > limit:
        if verbose:
            print(f"[cool] {t}C > {limit}C, sleeping 60s", flush=True)
        time.sleep(60)
        t = gpu_temp()


class SevereSet(Dataset):
    def __init__(self, root, size=512, rot=0):
        self.files = sorted(Path(root).glob("*.jpg"))
        assert self.files, f"no images in {root}"
        pre = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        if rot > 0:
            # 整图才有黑边可填；patch 数据旋转会造出假黑角，传 0 关掉
            pre.append(transforms.RandomRotation(rot))
        self.tf = transforms.Compose(pre + [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--noise_offset", type=float, default=0.1)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--temp_every", type=int, default=25)
    ap.add_argument("--temp_limit", type=int, default=70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--data", default=None, help="覆盖默认数据目录 b1b_data/severe512")
    ap.add_argument("--out", default=None, help="覆盖默认输出目录 b1b_lora")
    ap.add_argument("--rot", type=int, default=0, help="随机旋转角度上限（度）；patch 数据传 0")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda"
    data_dir = Path(a.data) if a.data else DATA
    out_dir = Path(a.out) if a.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] components", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(str(BASE / "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(str(BASE / "text_encoder"), torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(str(BASE / "vae"), torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(str(BASE / "unet"), torch_dtype=torch.float16).to(device)
    noise_sched = DDPMScheduler.from_pretrained(str(BASE / "scheduler"))

    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.train()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    print(f"[lora] rank={a.rank} alpha={a.alpha} trainable={sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)

    ds = SevereSet(data_dir, rot=a.rot)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    print(f"[data] {data_dir.name}: {len(ds)} images, {len(dl)} steps/epoch", flush=True)

    # 文本编码一次即可（固定 prompt）
    tok = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = text_encoder(tok.input_ids)[0]

    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler()

    start_step = 0
    if a.resume:
        from safetensors.torch import load_file
        sd = load_file(a.resume)
        missing, unexpected = unet.load_state_dict(sd, strict=False)
        print(f"[resume] loaded {a.resume} (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    step = start_step
    t0 = time.time()
    while step < a.steps:
        for batch in dl:
            if step >= a.steps:
                break
            x = batch.to(device, dtype=torch.float16)
            with torch.no_grad():
                lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(lat)
            if a.noise_offset > 0:
                noise += a.noise_offset * torch.randn(
                    (lat.shape[0], lat.shape[1], 1, 1), device=device, dtype=lat.dtype)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (lat.shape[0],),
                              device=device, dtype=torch.long)
            noisy = noise_sched.add_noise(lat, noise, t)

            emb_b = emb.expand(x.shape[0], -1, -1)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = unet(noisy, t, encoder_hidden_states=emb_b).sample
                loss = F.mse_loss(pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % 25 == 0:
                el = time.time() - t0
                print(f"[step {step}/{a.steps}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"temp={gpu_temp()}C elapsed={el/60:.1f}m", flush=True)
            if step % a.temp_every == 0:
                cool_wait(a.temp_limit)
            if step % a.save_every == 0 or step == a.steps:
                from safetensors.torch import save_file
                sd = {k: v.detach().clone() for k, v in unet.state_dict().items() if "lora" in k}
                fp = out_dir / f"lora_step_{step:05d}.safetensors"
                save_file(sd, str(fp))
                print(f"[save] {fp.name} ({len(sd)} tensors)", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
