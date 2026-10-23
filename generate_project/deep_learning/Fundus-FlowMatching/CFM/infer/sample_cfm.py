#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFM 采样入口（薄封装 → Fundus-Diffusion/ddpm/generate.py，自动 Euler）

用法
    python infer/sample_cfm.py --ckpt models/cfm256_final_model.pth \
        --num 8 --seed 42 --steps 50 --out judgment/cfm256_8
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CFM = HERE.parents[1]
DL = HERE.parents[3]
DDPM = DL / "Fundus-Diffusion" / "ddpm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--base_dim", type=int, default=128)
    ap.add_argument("--cond", default=str(CFM / "data/conditions_combined_256"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()

    cmd = [sys.executable, str(DDPM / "generate.py"),
           "--checkpoint", a.ckpt, "--num_images", str(a.num),
           "--seed", str(a.seed), "--sampling_steps", str(a.steps),
           "--img_size", str(a.img_size), "--base_dim", str(a.base_dim),
           "--cond_path", a.cond, "--output_dir", a.out, "--grid_size", "0",
           "--sampler", "auto"] + extra
    print(" ".join(cmd), flush=True)
    if a.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=str(DDPM)))


if __name__ == "__main__":
    main()
