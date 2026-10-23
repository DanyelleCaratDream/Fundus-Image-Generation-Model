#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交付批次：CFM@256 底图 → SDEdit 0.8（分批，防 60 张一次 OOM）

底图复用 eval_data/cfm256/singles（60 张，seed 100–159）；
输出 judgment_20261020/CFM256_delivery_s08/singles/，供作者挑选送审。
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

from train import UNet, GaussianDiffusion, check_environment
from generate import load_condition_images

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
DEFAULT_BASE = REPO / "eval_data" / "cfm256" / "singles"
DEFAULT_OUT = ROOT / "judgment_20261020" / "CFM256_delivery_s08" / "singles"


def load_images(files):
    out = []
    for f in files:
        a = torch.from_numpy(
            __import__("numpy").asarray(Image.open(f).convert("RGB"), dtype="float32") / 255.0
        ).permute(2, 0, 1)
        out.append(a * 2.0 - 1.0)
    return torch.stack(out)


@torch.no_grad()
def refine(model, diffusion, x1, t_start, steps, cond, align=False):
    b = x1.shape[0]
    x = t_start * x1 + (1.0 - t_start) * torch.randn_like(x1)
    dt = (1.0 - t_start) / steps
    for i in range(steps):
        t_cont = torch.full((b,), t_start + i * dt, device=x1.device)
        t_idx = diffusion.t_cont_to_index(t_cont, align=align)
        v = model(x, t_idx, cond=cond) if cond is not None else model(x, t_idx)
        x = x + v * dt
    return x.clamp(-1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT.parent.parent / "Fundus-FlowMatching/CFM/models/cfm256_final_model.pth"))
    ap.add_argument("--cond_path", default=str(ROOT / "conditions_combined_256"))
    ap.add_argument("--base_dir", default=str(DEFAULT_BASE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--base_dim", type=int, default=128)
    ap.add_argument("--t_start", type=float, default=0.8)
    ap.add_argument("--denoise_steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--noise_seed", type=int, default=1234)
    ap.add_argument("--no_cuda", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if check_environment(a.no_cuda) else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    ck_args = ckpt.get("args", None)
    use_ssn = getattr(ck_args, "use_scale_shift_norm", True) if ck_args is not None else True
    model = UNet(T=1000, channels=3, base_dim=a.base_dim, dim_mults=(1, 2, 3, 4),
                 attn_layers=(2,), dropout=0.0, cond_channels=1,
                 use_scale_shift_norm=use_ssn)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if "ema_state_dict" in ckpt and ckpt.get("has_ema", False):
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        print("[OK] EMA weights", flush=True)
    model.to(device).eval()

    all_files = sorted(Path(a.base_dir).glob("*.png"))
    assert all_files, f"no base images in {a.base_dir}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in out.glob("*.png")}
    todo = [(i, f) for i, f in enumerate(all_files) if f.name not in done]
    if done:
        print(f"[skip] {len(done)} 张已存在，续跑剩余 {len(todo)}", flush=True)
    conds_all = load_condition_images(a.cond_path, a.img_size, len(all_files)).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)

    print(f"[deliver] {len(todo)} bases, t_start={a.t_start}, chunk={a.chunk}", flush=True)
    for s in range(0, len(todo), a.chunk):
        part = todo[s:s + a.chunk]
        idxs = [i for i, _ in part]
        chunk = [f for _, f in part]
        x1 = load_images(chunk).to(device)
        conds = conds_all[idxs]
        torch.manual_seed(a.noise_seed + idxs[0])  # 按原始下标定种子，续跑可复现
        y = refine(model, diffusion, x1, a.t_start, a.denoise_steps, conds,
                   align=bool(getattr(ck_args, "cfm_time_align", False)))
        for j in range(len(chunk)):
            save_image(y[j], out / chunk[j].name, normalize=True)
        print(f"[ok] {len(done) + s + len(chunk)}/{len(all_files)}", flush=True)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
