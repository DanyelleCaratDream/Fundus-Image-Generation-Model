# -*- coding: utf-8 -*-
"""
pca_gen.py —— Phase C2 确定性基线 1：PCA 线性重建生成

方法原理：
    读训练图 → Resize 128x128 拉平成向量 → 中心化 + PCA 保留前 k 个主成分 →
    在主成分空间按特征值缩放采样高斯噪声 → 逆变换重建回像素图。
    本质是线性因子模型（低秩高斯生成模型），是 VAE 的最简退化版。

为什么作为基线：
    PCA 假设数据服从高斯线性子空间，但眼底图是强非高斯（黑背景/亮视盘/暗病灶）
    + 强结构（血管走向）。线性假设把一切拉向"平均脸"，病灶被平均掉。
    预期结果（否定性）：结构全无、色彩偏"平均脸"、血管/病灶全部溶解。

文献依据：
    report/01-literature-trad-ml.md §1.1（Eigenfaces: Turk & Pentland 1991）
    report/01-literature-trad-ml.md §6.1（COCA 指出 PCA 非高斯伪影）

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/pca_gen.py --num_images 300 --seed 42

输出：../../eval_data/pca/singles/sample_0000.png ...（对齐评估口径）
"""
import argparse
import os
import glob

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


def load_images(folder, img_size=128):
    """加载全部素材图，返回 [N, H*W*C] float32（0-255）。"""
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.jpeg")) +
                   glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise FileNotFoundError(f"目录无图像: {folder}")
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float32).reshape(-1))
    return np.stack(imgs)


def main():
    ap = argparse.ArgumentParser(description="PCA 线性重建生成基线（确定性基线 1）")
    ap.add_argument("--data", default="../../fundus/_all_images_ORIGINAL", help="素材图目录")
    ap.add_argument("--output", default="../../eval_data/pca", help="输出根目录")
    ap.add_argument("--img_size", type=int, default=128, help="生成分辨率")
    ap.add_argument("--k", type=int, default=64, help="保留主成分个数")
    ap.add_argument("--num_images", type=int, default=300, help="生成数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    np.random.seed(args.seed)
    D = args.img_size * args.img_size * 3
    print(f"[PCA] seed={args.seed} 加载素材: {args.data}")
    X = load_images(args.data, args.img_size)
    print(f"[PCA] 素材 {X.shape[0]} 张，每张 {D} 维")

    # 中心化 + PCA（白化：每主成分按特征值归一，采样后 inverse 还原值域）
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    pca = PCA(n_components=args.k, whiten=True, random_state=args.seed)
    pca.fit(Xc)
    print(f"[PCA] 已拟合 k={args.k}（累计方差 {pca.explained_variance_ratio_.sum() * 100:.1f}%）")

    # 生成：标准正态采样 → 逆白化还原到图像空间
    rng = np.random.RandomState(args.seed)
    Z = rng.standard_normal((args.num_images, args.k))
    X_gen = pca.inverse_transform(Z) + mean
    print(f"[PCA] 已生成 {args.num_images} 张潜空间采样")

    singles = os.path.join(args.output, "singles")
    os.makedirs(singles, exist_ok=True)
    for i in range(args.num_images):
        img = X_gen[i].reshape(args.img_size, args.img_size, 3)
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(singles, f"sample_{i:04d}.png"))
    print(f"[PCA] 已保存到 {singles}")


if __name__ == "__main__":
    main()
