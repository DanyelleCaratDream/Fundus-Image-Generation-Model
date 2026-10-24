#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索条件模型的采样 + 评估（多样性 + B 域指标）

用法
    python tools/eval_feat_model.py --ckpt results/cfm128_featC/models/final_model.pth \
        --conds data/handcrafted/conds_severe.npy --tag cfm128_featC --num 60
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

CFM = Path(__file__).resolve().parents[1]
DL = CFM.parents[1]
REPO = DL.parents[1]
sys.path.insert(0, str(DL / "Fundus-Diffusion" / "ddpm"))
from train import UNet, GaussianDiffusion  # noqa: E402
from generate import load_condition_images  # noqa: E402


class FeatWrap:
    def __init__(self, model, vec):
        self.model, self.vec = model, vec

    def __call__(self, x, t, cond=None):
        v = self.vec[None].expand(x.shape[0], -1).to(x.device)
        return self.model(x, t, cond=cond, text_emb=v)


def build(ckpt, base_dim, feat_dim):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ca = ck.get("args", None)
    ssn = getattr(ca, "use_scale_shift_norm", True) if ca else True
    model = UNet(T=1000, channels=3, base_dim=base_dim, dim_mults=(1, 2, 3, 4),
                 attn_layers=(2,), dropout=0.0, cond_channels=1,
                 use_scale_shift_norm=ssn, text_dim=feat_dim,
                 use_time_cond=not bool(getattr(ca, "no_time_cond", False)) if ca else True)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    if "ema_state_dict" in ck and ck.get("has_ema", False):
        model.load_state_dict(ck["ema_state_dict"], strict=False)
    return model.cuda().eval(), ca


def diversity(files, n_pairs=400, seed=0):
    import lpips
    fn = lpips.LPIPS(net="alex").cuda().eval()
    imgs = []
    for f in files:
        a = Image.open(f).convert("RGB").resize((128, 128))
        imgs.append(torch.from_numpy(np.asarray(a, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1))
    x = torch.stack(imgs).cuda()
    rng = np.random.default_rng(seed)
    vals = []
    with torch.no_grad():
        for _ in range(n_pairs):
            i, j = rng.integers(0, len(x), 2)
            if i == j:
                continue
            vals.append(float(fn(x[i:i+1], x[j:j+1]).item()))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--base_dim", type=int, default=128)
    ap.add_argument("--cond", default=str(CFM / "data/conditions_combined_128"))
    ap.add_argument("--out_root", default=str(REPO / "eval_data"))
    ap.add_argument("--skip_metrics", action="store_true")
    a = ap.parse_args()

    feat_dir = Path(a.conds).parent
    feat_dim = int(np.load(feat_dir / "feats.npy", mmap_mode="r").shape[1])
    model, ca = build(a.ckpt, a.base_dim, feat_dim)
    time_align = bool(getattr(ca, "cfm_time_align", False)) if ca else False
    conds = torch.from_numpy(np.load(a.conds)).float().cuda()
    cond_imgs = load_condition_images(a.cond, a.img_size, a.num).cuda()
    diffusion = GaussianDiffusion(timesteps=1000).cuda()

    out = Path(a.out_root) / a.tag / "singles"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.png"))) < a.num:
        with torch.no_grad():
            for i in range(a.num):
                torch.manual_seed(a.seed + i)
                wrap = FeatWrap(model, conds[i % len(conds)])
                y = diffusion.euler_sample(wrap, (1, 3, a.img_size, a.img_size), "cuda",
                                           sampling_steps=a.steps, cond=cond_imgs[i:i+1],
                                           time_align=time_align)
                save_image(y[0], out / f"sample_{i}.png", normalize=True)
                if (i + 1) % 10 == 0:
                    print(f"[gen] {i+1}/{a.num}", flush=True)
    files = sorted(out.glob("*.png"))
    print(f"[data] {len(files)} images -> {out}")

    div = diversity(files)
    res = {"tag": a.tag, "n": len(files), "lpips_pairwise": div}
    if (REPO / "eval_data/cfm128/singles").exists():
        ref = sorted((REPO / "eval_data/cfm128/singles").glob("*.png"))[:len(files)]
        res["ref_cfm128_lpips_pairwise"] = diversity(ref)
    (Path(a.out_root) / f"{a.tag}_diversity.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[diversity]", res)

    if a.skip_metrics:
        return
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
    py = sys.executable
    subprocess.run([py, str(REPO / "eval/metrics_common.py"),
                    "--real", str(REPO / "eval_data/real_severe_train"),
                    "--fake", str(out), "--img_size", "256", "--device", "cuda", "--json"],
                   cwd=str(REPO), stdout=open(REPO / f"eval_data/{a.tag}_metrics.json", "w"),
                   stderr=subprocess.DEVNULL, env=env)
    subprocess.run([py, str(REPO / "eval/metrics_fundus.py"),
                    "--real", str(REPO / "eval_data/real_severe_train"),
                    "--fake", str(out), "--model", a.tag, "--img_size", "256",
                    "--device", "cuda", "--cond_path", str(CFM / "data/conditions_combined_256"),
                    "--json"],
                   cwd=str(REPO), stdout=open(REPO / f"eval_data/{a.tag}_fundus_metrics.json", "w"),
                   stderr=subprocess.DEVNULL, env=env)
    print(f"[done] metrics -> eval_data/{a.tag}_*_metrics.json")


if __name__ == "__main__":
    main()
