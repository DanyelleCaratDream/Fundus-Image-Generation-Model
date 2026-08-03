# -*- coding: utf-8 -*-
"""
patch_gen.py —— Phase C2 确定性基线 3：补丁拼接（无结构重组）

方法原理：
    读训练图 → Resize 128x128 → 从随机训练图上随机裁 n_per_side^2 个 patch →
    拼成一张新图（块之间不做平滑/对齐）。

为什么作为基线：
    纯数据重排、无任何生成模型。用来直观验证"随机拼接没有解剖结构"，
    提供最底层对照。同时它体现补丁法的"局部复制"陷阱（拼布=复制原图局部），
    是复制检测三件套要重点盯的方法。

文献依据：
    report/02-literature-fundus.md §4（补丁法局部复制风险）
    report/01-literature-trad-ml.md §3.2（Image Quilting: Efros & Freeman 2001，
    但本基线不做 min-cut 接缝，是无缝拼接的最朴素版）

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/patch_gen.py --num_images 300 --seed 42

输出：../../eval_data/patch/singles/sample_0000.png ...
"""
import argparse
import os
import glob

import numpy as np
from PIL import Image


def load_images(folder, img_size=128):
    """加载全部素材图，返回 [N, H, W, C] uint8（0-255）。"""
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.jpeg")) +
                   glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise FileNotFoundError(f"目录无图像: {folder}")
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.uint8))
    return np.stack(imgs)


def main():
    ap = argparse.ArgumentParser(description="补丁拼接生成基线（确定性基线 3，无模型纯数据重排）")
    ap.add_argument("--data", default="../../fundus/_all_images_ORIGINAL", help="素材图目录")
    ap.add_argument("--output", default="../../eval_data/patch", help="输出根目录")
    ap.add_argument("--img_size", type=int, default=128, help="生成分辨率")
    ap.add_argument("--patch_size", type=int, default=64, help="拼接块边长（须整除 img_size）")
    ap.add_argument("--num_images", type=int, default=300, help="生成数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    print(f"[PATCH] seed={args.seed} 加载素材: {args.data}")
    imgs = load_images(args.data, args.img_size)
    N = imgs.shape[0]
    if args.img_size % args.patch_size != 0:
        raise ValueError("patch_size 必须整除 img_size")
    n_per_side = args.img_size // args.patch_size
    print(f"[PATCH] 素材 {N} 张，每张拼 {n_per_side}x{n_per_side} 块")

    singles = os.path.join(args.output, "singles")
    os.makedirs(singles, exist_ok=True)
    for i in range(args.num_images):
        canvas = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
        for r in range(n_per_side):
            for c in range(n_per_side):
                # 随机选一张图 + 随机起点（保证块在边界内）
                src = imgs[rng.randint(N)]
                max_start = args.img_size - args.patch_size
                y0 = rng.randint(0, max_start + 1)
                x0 = rng.randint(0, max_start + 1)
                patch = src[y0:y0 + args.patch_size, x0:x0 + args.patch_size]
                canvas[r * args.patch_size:(r + 1) * args.patch_size,
                       c * args.patch_size:(c + 1) * args.patch_size] = patch
        Image.fromarray(canvas).save(os.path.join(singles, f"sample_{i:04d}.png"))
    print(f"[PATCH] 已生成 {args.num_images} 张并保存到 {singles}")


if __name__ == "__main__":
    main()
