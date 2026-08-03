# -*- coding: utf-8 -*-
"""
gmm_gen.py —— Phase C2 确定性基线 2：GMM 高斯混合采样

方法原理：
    读训练图 → Resize 128x128 → 先 PCA 降到 pca_dim 维（控制 GMM 维数，避免稀疏）→
    在降维特征空间拟合 GaussianMixture（K 个分量）→ 从 GMM 采样 → 逆 PCA 还原。

为什么作为基线：
    GMM 直接对像素建模必败（维度灾难 + 无空间关系）。在紧凑表示（潜空间）上可行，
    但只能捕获聚类中心附近的"平均形态"，无法生成锐利结构。
    预期结果（否定性）：与 PCA 类似但更"聚类化"，可能重复某些典型形态。

文献依据：
    report/01-literature-trad-ml.md §2.1（GMM 点云/神经影像: Yang 2019）
    report/01-literature-trad-ml.md §2.1（lab2im 条件 GMM 生成脑 MRI）

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/gmm_gen.py --num_images 300 --seed 42

输出：../../eval_data/gmm/singles/sample_0000.png ...
"""
import argparse
import os
import glob

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


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
    ap = argparse.ArgumentParser(description="GMM 高斯混合采样生成基线（确定性基线 2）")
    ap.add_argument("--data", default="../../fundus/_all_images_ORIGINAL", help="素材图目录")
    ap.add_argument("--output", default="../../eval_data/gmm", help="输出根目录")
    ap.add_argument("--img_size", type=int, default=128, help="生成分辨率")
    ap.add_argument("--pca_dim", type=int, default=64, help="先 PCA 降到该维再拟合 GMM")
    ap.add_argument("--n_components", type=int, default=16, help="GMM 混合分量数 K")
    ap.add_argument("--num_images", type=int, default=300, help="生成数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    np.random.seed(args.seed)
    D = args.img_size * args.img_size * 3
    print(f"[GMM] seed={args.seed} 加载素材: {args.data}")
    X = load_images(args.data, args.img_size)
    print(f"[GMM] 素材 {X.shape[0]} 张，每张 {D} 维")

    # 第一步：PCA 降维到 pca_dim（去相关 + 降维）
    mean = X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=args.pca_dim, whiten=True, random_state=args.seed)
    F = pca.fit_transform(X - mean)  # [N, pca_dim]
    print(f"[GMM] PCA 降维到 {args.pca_dim} 维（累计方差 "
          f"{pca.explained_variance_ratio_.sum() * 100:.1f}%）")

    # 第二步：在降维特征上拟合 GMM
    gmm = GaussianMixture(n_components=args.n_components, covariance_type="full",
                          random_state=args.seed, reg_covar=1e-4, max_iter=300)
    gmm.fit(F)
    print(f"[GMM] GMM 拟合完成：K={args.n_components}, BIC={gmm.bic(F):.0f}")

    # 第三步：从 GMM 采样 → 逆 PCA 还原
    rng = np.random.RandomState(args.seed)
    Z, _ = gmm.sample(args.num_images)  # [N, pca_dim]
    X_gen = pca.inverse_transform(Z) + mean
    print(f"[GMM] 已生成 {args.num_images} 张混合采样")

    singles = os.path.join(args.output, "singles")
    os.makedirs(singles, exist_ok=True)
    for i in range(args.num_images):
        img = X_gen[i].reshape(args.img_size, args.img_size, 3)
        img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(singles, f"sample_{i:04d}.png"))
    print(f"[GMM] 已保存到 {singles}")


if __name__ == "__main__":
    main()
