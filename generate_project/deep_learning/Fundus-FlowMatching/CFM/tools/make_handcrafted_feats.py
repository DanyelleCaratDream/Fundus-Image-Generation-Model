#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""手工可解释特征（C 路编码器）：颜色 / 血管 / 病灶样统计，供检索+重心条件

每个特征都有明确含义，便于论文解释：
  rgb_mean_*, rgb_std_*     盘内 RGB 均值/标准差（色彩与对比度）
  lab_mean_*                盘内 LAB 均值（亮度/色度）
  vessel_frac               血管 mask 覆盖率（血管密度）
  yellow_frac               黄白（渗出样）像素占比
  darkred_frac              暗红（出血样）像素占比
  bright_spot_count         局部亮斑数量（top-hat 连通块）
  dark_spot_count           局部暗斑数量
  contrast                  盘内亮度标准差（整体对比度）

输出：CFM/data/handcrafted/feats.npy + names.json

用法
    python tools/make_handcrafted_feats.py --src ../../../fundus/combined_sq256 \
        --cond data/conditions_combined_256 --out data/handcrafted
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def disk_mask(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= (0.47 * min(h, w)) ** 2


def feat_one(img_bgr, vessel):
    h, w = img_bgr.shape[:2]
    m = disk_mask(h, w)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    def mean_std(x):
        v = x[m]
        return float(v.mean()), float(v.std())

    feats = []
    for c in range(3):
        mu, sd = mean_std(rgb[..., c]); feats += [mu, sd]
    for c in range(3):
        feats.append(float(lab[..., c][m].mean()))

    # 血管密度（条件 mask）
    if vessel is not None:
        vm = (vessel > 60) & m
        feats.append(float(vm.sum() / max(1, m.sum())))
    else:
        feats.append(0.0)

    # 病灶样颜色
    yellow = (r > 0.55) & (g > 0.38) & (b < 0.45) & ((r - b) > 0.15) & m
    darkred = ((r - g) > 0.12) & (g < 0.32) & (lum < 0.36) & m
    feats += [float(yellow.sum() / m.sum()), float(darkred.sum() / m.sum())]

    # 亮/暗斑点数量（top-hat / black-hat）
    gray = (lum * 255).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker)
    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
    n_bright = cv2.connectedComponents((top > 25).astype(np.uint8), 8)[0]
    n_dark = cv2.connectedComponents((black > 25).astype(np.uint8), 8)[0]
    feats += [float(max(0, n_bright - 1)) / 100.0, float(max(0, n_dark - 1)) / 100.0]

    # 整体对比度
    feats.append(float(lum[m].std()))
    return np.array(feats, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cond", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted([p for p in Path(a.src).iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    feats, names = [], []
    for i, f in enumerate(files):
        img = cv2.resize(cv2.imread(str(f)), (256, 256), interpolation=cv2.INTER_AREA)
        vessel = None
        if a.cond:
            vp = Path(a.cond) / (f.stem + ".png")
            if vp.exists():
                vessel = cv2.resize(cv2.imread(str(vp), 0), (256, 256))
        feats.append(feat_one(img, vessel)); names.append(f.name)
        if i % 40 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    feats = np.stack(feats)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "feats.npy", feats)
    (out / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {feats.shape} -> {out}")
    print("  feature ranges:", np.round(feats.min(0), 3).tolist(), "..", np.round(feats.max(0), 3).tolist())


if __name__ == "__main__":
    main()
