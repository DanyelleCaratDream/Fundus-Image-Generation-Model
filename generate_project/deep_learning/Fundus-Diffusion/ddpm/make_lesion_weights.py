#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成病灶权重图（供 train.py --lesion_weight_dir 使用）

思路（无像素级标注，用色彩/形态启发式）：
- 渗出：亮黄（R/G 高、B 低、亮度高），面积 8–400 px 的连通块
- 出血：暗红（R−G 大、G 低、亮度低），面积 6–300 px 的连通块
- 排除视盘：最大的高亮连通块（面积 > 600 且近似圆形）从渗出里剔除
- 输出：0=非病灶，255=病灶中心；膨胀 3 + 高斯模糊 σ=2 做软边

用法
    F:/python/python.exe make_lesion_weights.py \
        --src ../../../../fundus/combined_sq256 --out ./lesion_weights_256 --size 256
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def masks(img_bgr, vessel=None):
    """vessel: 血管 mask（0-255，来自 conditions_combined_*）；用于排除血管假阳性"""
    b, g, r = [c.astype(np.float32) / 255.0 for c in cv2.split(img_bgr)]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # 只做渗出（金色亮斑）：实测比出血启发式可靠得多；出血/血管难区分，放弃
    m = ((r > 0.55) & (g > 0.38) & (b < 0.45) & ((r - b) > 0.15)).astype(np.uint8)

    # 减血管（骨架膨胀 5px）：血管边缘是最主要的假阳性来源
    if vessel is not None:
        vm = cv2.dilate((vessel > 60).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        m[vm > 0] = 0

    # 连通块：面积 + 实心度（排除细长/噪声）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 20 or a > 600:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        ha = max(1.0, cv2.contourArea(cv2.convexHull(cnts[0])))
        if a / ha >= 0.50:
            out[lab == i] = 1

    if out.sum() == 0:
        return np.zeros((out.shape[0], out.shape[1]), np.uint8)
    soft = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 2.0)
    # 膨胀式羽化：覆盖病灶周边，放大加权区域（纯色块太小对 loss 影响有限）
    soft = cv2.dilate(soft, np.ones((7, 7), np.float32), iterations=1)
    soft = cv2.GaussianBlur(soft, (0, 0), 3.0)
    soft = np.clip(soft / max(1e-6, float(soft.max())), 0, 1)
    # 覆盖率上限：>15% 视为误检（大面积偏色），清零避免错误加权
    if (soft > 0.3).mean() > 0.15:
        return np.zeros((soft.shape[0], soft.shape[1]), np.uint8)
    return (soft * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cond", default=None, help="血管 mask 目录（conditions_combined_256）")
    a = ap.parse_args()

    files = sorted([p for p in Path(a.src).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    assert files, f"no images in {a.src}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cover = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_LANCZOS4)
        vessel = None
        if a.cond:
            vp = Path(a.cond) / (f.stem + ".png")
            if vp.exists():
                vessel = cv2.imread(str(vp), cv2.IMREAD_GRAYSCALE)
                if vessel is not None:
                    vessel = cv2.resize(vessel, (a.size, a.size))
        m = masks(img, vessel)
        cv2.imwrite(str(out / (f.stem + ".png")), m)
        cover.append((m > 32).mean())
    cover = np.array(cover)
    print(f"[done] {len(files)} -> {out}")
    print(f"[stats] 病灶覆盖率 mean={cover.mean():.3%} median={np.median(cover):.3%} "
          f"max={cover.max():.3%}  (零覆盖 {int((cover < 0.001).sum())} 张)")


if __name__ == "__main__":
    main()
