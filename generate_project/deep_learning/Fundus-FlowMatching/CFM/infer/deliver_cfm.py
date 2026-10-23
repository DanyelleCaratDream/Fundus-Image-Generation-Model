#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交付批次入口（薄封装 → ddpm/deliver_cfm256_s08.py，SDEdit 0.8 分批 + 续跑）

用法
    python infer/deliver_cfm.py --base <底图目录> --out judgment/delivery_s08 --chunk 4
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
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(CFM / "results/cfm256/models/final_model.pth"))
    ap.add_argument("--cond", default=str(CFM / "data/conditions_combined_256"))
    ap.add_argument("--t_start", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()

    cmd = [sys.executable, str(DDPM / "deliver_cfm256_s08.py"),
           "--checkpoint", a.ckpt, "--cond_path", a.cond,
           "--base_dir", a.base, "--out", a.out,
           "--t_start", str(a.t_start), "--denoise_steps", str(a.steps),
           "--chunk", str(a.chunk)] + extra
    print(" ".join(cmd), flush=True)
    if a.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=str(DDPM)))


if __name__ == "__main__":
    main()
