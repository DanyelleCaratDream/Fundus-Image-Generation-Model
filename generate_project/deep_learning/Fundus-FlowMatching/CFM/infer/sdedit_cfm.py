#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFM 自精修入口（薄封装 → ddpm/f3_cfm256_sdedit.py，原生 256 默认）

CFM 版 SDEdit：x_t = t_start·x1 + (1−t_start)·noise，Euler 从 t_start 积到 1。
等价档位：DDPM t0=200 ↔ t_start=0.8；t0=300 ↔ 0.7。

用法
    python infer/sdedit_cfm.py --ckpt models/cfm256_final_model.pth \
        --t_start 0.8 --out judgment/cfm256_s08
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
    ap.add_argument("--ckpt", default=str(CFM / "models/cfm256_final_model.pth"))
    ap.add_argument("--cond", default=str(CFM / "data/conditions_combined_256"))
    ap.add_argument("--t_start", type=float, nargs="+", default=[0.8])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--base", default=str(DL / "Fundus-Diffusion/ddpm/judgment_20261020/D5_cfm256_ema/singles"))
    ap.add_argument("--out", default=str(CFM / "judgment/cfm256_sdedit"))
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()

    cmd = [sys.executable, str(DDPM / "f3_cfm256_sdedit.py"),
           "--checkpoint", a.ckpt, "--cond_path", a.cond,
           "--t_start", *[str(x) for x in a.t_start],
           "--denoise_steps", str(a.steps), "--num_images", str(a.num),
           "--base_dir", a.base, "--out_root", a.out] + extra
    print(" ".join(cmd), flush=True)
    if a.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=str(DDPM)))


if __name__ == "__main__":
    main()
