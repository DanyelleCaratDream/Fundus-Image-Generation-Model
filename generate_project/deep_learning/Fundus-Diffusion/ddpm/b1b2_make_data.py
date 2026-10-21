#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路线 A 数据准备：高清 severe 源图 → 512 patch 裁剪（b1b_data/severe_patches）

动机（EXPERIMENTS 执行记录 8/9）：72 张整图 resize 让视盘霸占画面中央、病灶扎堆视盘旁；
52% 源图 ≥1536px 却被压到 512。本脚本从高清方图裁 512×512 patch：
- 每张大图（短边 ≥700px）1 张中心 patch（含视盘）+ 3 张随机 patch（圆盘内、前景占比 ≥0.55）
- 小图退化为整图 resize
跳过 _aug_1；固定 seed=42 可复现。b1b_data/ 不入 git。

用法
    F:/python/python.exe b1b2_make_data.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(r"D:\AI_Model_Project_for_Fundus_Color_Images\fundus\HRSG_4_758_square\train\severe")
OUT = Path(__file__).resolve().parent / "b1b_data" / "severe_patches"
PATCH = 512
PER_BIG = 4
MIN_SIDE = 700
SEED = 42


def fg_frac(arr):
    """前景（眼底圆盘）像素占比"""
    return (arr.mean(axis=2) > 0.12 * 255).mean()


def main():
    rng = np.random.default_rng(SEED)
    files = sorted(f for f in SRC.glob("*.jpg") if "_aug_1" not in f.name)
    assert files, f"no source images in {SRC}"
    OUT.mkdir(parents=True, exist_ok=True)
    n_big = n_small = 0
    for f in files:
        img = Image.open(f).convert("RGB")
        stem = f.stem
        if min(img.size) < MIN_SIDE:
            img.resize((PATCH, PATCH), Image.LANCZOS).save(OUT / f"{stem}_whole.jpg", quality=95)
            n_small += 1
            continue
        n_big += 1
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        r_fundus = 0.5 * min(w, h)
        # 中心 patch（含视盘区域）
        cx, cy = w // 2, h // 2
        img.crop((cx - PATCH // 2, cy - PATCH // 2,
                  cx + PATCH // 2, cy + PATCH // 2)).save(OUT / f"{stem}_c.jpg", quality=95)
        # 随机 patch：中心落在眼底圆内（留出 patch 半径），前景 ≥0.55 才收
        for k in range(PER_BIG - 1):
            rmax = max(0.0, (r_fundus - PATCH / 2) * 0.8)
            for tries in range(40):
                rr = rng.uniform(0, rmax) if rmax > 0 else 0.0
                th = rng.uniform(0, 2 * np.pi)
                px = int(w / 2 + rr * np.cos(th))
                py = int(h / 2 + rr * np.sin(th))
                x0 = int(np.clip(px - PATCH // 2, 0, w - PATCH))
                y0 = int(np.clip(py - PATCH // 2, 0, h - PATCH))
                patch = arr[y0:y0 + PATCH, x0:x0 + PATCH]
                if fg_frac(patch) >= 0.55:
                    break
            Image.fromarray(patch).save(OUT / f"{stem}_p{k}.jpg", quality=95)
    total = len(list(OUT.glob("*.jpg")))
    print(f"[done] 源图 {len(files)}（大 {n_big} / 小 {n_small}）→ {total} patches -> {OUT}")


if __name__ == "__main__":
    main()
