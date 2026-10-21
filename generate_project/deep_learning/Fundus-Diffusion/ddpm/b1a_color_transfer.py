#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DL-EX-025 B1a 色彩后处理：Reinhard LAB 迁移

锚点 = 真实 B 域 train split（老师数据集 HRSG_4_758_sq256/train，545 张）的圆形区域内统计，
不是 D4 生成图（避免二手偏差）。

只统计眼底圆盘内像素（亮度阈值），避免黑边拉偏均值/方差。
输出：judgment_20261020/B1a_lora_ct/singles/ + 对比图 b1a_ct_compare.png
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REAL_DIR = Path(r"D:\AI_Model_Project_for_Fundus_Color_Images\fundus\HRSG_4_758_sq256\train")
SRC_DIR = ROOT / "judgment_20261020" / "B1a_lora" / "singles"
OUT_DIR = ROOT / "judgment_20261020" / "B1a_lora_ct" / "singles"


def fundus_mask(arr):
    """眼底前景 mask：中心圆盘（半径 0.47·min(H,W)）∩ 亮度阈值。

    生成图的背景不是纯黑（是灰色），单靠亮度会把背景算进前景，必须在几何上先限死。
    """
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = 0.47 * min(h, w)
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    lum = arr.mean(axis=2)
    m = disk & (lum > 0.12)
    if m.sum() < 1000:
        m = disk
    return m


def lab_stats(img: Image.Image):
    """返回圆盘内 LAB 各通道均值/标准差"""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    lab = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).convert("LAB"), dtype=np.float32)
    m = fundus_mask(arr)
    flat = lab[m]
    return flat.mean(axis=0), flat.std(axis=0)


def collect_real_stats():
    files = sorted(REAL_DIR.rglob("*.jpg"))
    ms, ss = [], []
    for f in files:
        try:
            mu, sd = lab_stats(Image.open(f))
            ms.append(mu); ss.append(sd)
        except Exception:
            continue
    ms = np.stack(ms); ss = np.stack(ss)
    print(f"[real] {len(ms)} images, LAB mean={ms.mean(0).round(1)}, std={ss.mean(0).round(1)}")
    return ms.mean(0), ss.mean(0)


def transfer(src: Image.Image, mu_t, sd_t):
    arr = np.asarray(src.convert("RGB"), dtype=np.float32) / 255.0
    lab = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).convert("LAB"), dtype=np.float32)
    m = fundus_mask(arr)
    flat = lab[m]
    mu_s, sd_s = flat.mean(axis=0), flat.std(axis=0)
    sd_s = np.maximum(sd_s, 1e-3)
    # L 通道完整对齐（修亮度/对比度）；a/b 只平移色温、锁方差，
    # 否则真实域更大的色彩方差会把生成图的局部色偏放大成高饱和色块
    scale = np.array([sd_t[0] / sd_s[0], 1.0, 1.0], dtype=np.float32)
    out = lab.copy()
    out[m] = (flat - mu_s) * scale + mu_t
    out[m] = np.clip(out[m], 0, 255)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="LAB").convert("RGB")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-compare", action="store_true", help="跳过内置对比图")
    a = ap.parse_args()
    src_dir, out_dir = Path(a.src), Path(a.out)

    out_dir.mkdir(parents=True, exist_ok=True)
    mu_t, sd_t = collect_real_stats()
    srcs = sorted(src_dir.glob("*.png"))
    print(f"[src] {len(srcs)} images")
    for f in srcs:
        im = Image.open(f)
        out = transfer(im, mu_t, sd_t)
        out.save(out_dir / f.name)
        print(f"[ok] {f.name}")
    if a.no_compare:
        print("[done] skip compare sheet")
        return
    # 对比图：上原始 / 下迁移后，各 4 张 512 缩到 320
    from PIL import ImageDraw
    S, PAD, CAP = 320, 6, 24
    n = min(4, len(srcs))
    canvas = Image.new("RGB", (n * S + (n + 1) * PAD, 2 * (S + CAP + PAD) + PAD), "white")
    dr = ImageDraw.Draw(canvas)
    y = PAD
    for label, d in [("B1a original (LoRA, 512)", src_dir), ("B1a + Reinhard LAB transfer to real B-domain train stats", out_dir)]:
        dr.text((PAD, y), label, fill="black"); y += CAP
        for i, f in enumerate(sorted(d.glob("*.png"))[:n]):
            canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC), (PAD + i * (S + PAD), y))
        y += S + PAD
    out = ROOT / "judgment_20261020" / "b1a_ct_compare.png"
    canvas.save(out)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
