#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查点扫描（诊断用）：同种子跨检查点出图 + 廉价代理 + 并排图

**只作诊断，不作为主结果选择**（见 ROADMAP/计划附录：主结果固定用 final EMA，
避免与 D3/D4 等只有 final 的方法不可比，也避免测试集选择偏差）。

用法
    # 普通 CFM（cfm128）
    python tools/scan_checkpoints.py --ckpt_dir "D:/tmp/_archive_cfm_ckpts_20261023/cfm128" \
        --epochs 800 900 1000 1100 --img_size 128 --cond data/conditions_combined_128 --tag cfm128

    # 检索条件模型（featC/featA）
    python tools/scan_checkpoints.py --ckpt_dir results/cfm128_featC/models \
        --epochs 800 900 1000 1100 --img_size 128 --cond data/conditions_combined_128 \
        --feat_conds data/handcrafted/conds_severe.npy --tag cfm128_featC
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CFM = Path(__file__).resolve().parents[1]
DL = CFM.parents[1]
REPO = DL.parents[1]
DDPM = DL / "Fundus-Diffusion" / "ddpm"
PY = sys.executable


def sh(cmd, cwd):
    print("RUN", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd)).returncode


def stats(files, real_mean=None):
    import torch
    try:
        import lpips
        fn = lpips.LPIPS(net="alex").cuda().eval()
    except Exception:
        fn = None
    arrs, tens = [], []
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        arrs.append(a)
        tens.append(torch.from_numpy(a * 2 - 1).permute(2, 0, 1))
    mu = np.mean([a.mean((0, 1)) for a in arrs], axis=0) * 255
    white = float(np.mean([(a > 0.95).mean() for a in arrs]))
    div = None
    if fn is not None and len(tens) >= 2:
        x = torch.stack(tens).cuda()
        vals = []
        with torch.no_grad():
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    vals.append(float(fn(x[i:i+1], x[j:j+1]).item()))
        div = float(np.mean(vals))
    return {"rgb_mean": [round(float(v), 1) for v in mu], "white_rate": round(white, 4),
            "lpips_pairwise": round(div, 4) if div is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--feat_conds", default=None)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else CFM / "results" / a.tag / "scan"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ep in a.epochs:
        ck = Path(a.ckpt_dir) / f"checkpoint_epoch_{ep:05d}.pth"
        if not ck.exists():
            print(f"[skip] missing {ck}")
            continue
        d = out / f"epoch_{ep:05d}"
        d.mkdir(parents=True, exist_ok=True)
        if a.feat_conds:
            cmd = [PY, CFM / "tools/sample_feat.py", "--ckpt", ck, "--conds", a.feat_conds,
                   "--num", a.num, "--seed", a.seed, "--steps", a.steps,
                   "--img_size", a.img_size, "--cond_path", a.cond, "--out", d]
            sh(cmd, CFM)
        else:
            cmd = [PY, DDPM / "generate.py", "--checkpoint", ck, "--num_images", a.num,
                   "--img_size", a.img_size, "--base_dim", "128", "--seed", a.seed,
                   "--sampling_steps", a.steps, "--output_dir", d, "--grid_size", "0",
                   "--cond_path", a.cond]
            sh(cmd, DDPM)
        files = sorted(Path(d).glob("*.png"))
        if files:
            summary[ep] = stats(files)
            print(f"[ok] {ep}: {summary[ep]}", flush=True)

    (out / "scan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    # 并排图
    rows = sorted(summary)
    if rows:
        S, PAD, CAP, N = 200, 5, 20, a.num
        canvas = Image.new("RGB", (N * S + (N + 1) * PAD, len(rows) * (S + CAP + PAD) + PAD), "white")
        dr = ImageDraw.Draw(canvas); y = PAD
        for ep in rows:
            dr.text((PAD, y), f"epoch {ep}", fill="black"); y += CAP
            for i, f in enumerate(sorted((out / f"epoch_{ep:05d}").glob("*.png"))[:N]):
                canvas.paste(Image.open(f).convert("RGB").resize((S, S)), (PAD + i * (S + PAD), y))
            y += S + PAD
        canvas.save(out / f"scan_{a.tag}.png")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
