"""
模型评估脚本
=============
计算生成图像与真实图像之间的 FID（Fréchet Inception Distance）。
FID 越低表示生成图像越接近真实图像分布。

用法:
    # 评估 VAE 生成结果
    python evaluate.py --real_path "../fundus/_all_images_ORIGINAL" \\
        --fake_path "../Fundus-VAE/results/generated/singles" \\
        --img_size 128

    # 批量评估所有模型
    python evaluate.py --real_path "../fundus/_all_images_ORIGINAL" \\
        --fake_path "../Fundus-GAN/dcgan/results/generated/singles"
"""

import argparse
import os
import sys
import glob
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import inception_v3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated fundus images with FID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real_path", type=str, required=True, help="真实图片文件夹路径")
    parser.add_argument("--fake_path", type=str, required=True, help="生成图片文件夹路径")
    parser.add_argument("--img_size", type=int, default=128, help="图片分辨率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    return parser.parse_args()


def load_images_from_folder(folder, img_size=128, max_images=None):
    """加载文件夹中的所有图片，返回归一化后的 tensor。"""
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))

    paths = sorted(list(set(paths)))
    if max_images:
        paths = paths[:max_images]

    if len(paths) == 0:
        print(f"[WARN] 在 {folder} 中未找到图片")
        return None

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = transform(img)
        images.append(img)

    return torch.stack(images)


def get_inception_features(images, model, device, batch_size=32):
    """用 InceptionV3 提取图像特征。"""
    n = len(images)
    features = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = images[i:i+batch_size].to(device)
            # InceptionV3 需要 299x299 输入，先调整大小
            if batch.shape[-1] != 299:
                batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode="bilinear")
            feat = model(batch)
            features.append(feat.cpu())

    return torch.cat(features, dim=0)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """计算 FID 分数。"""
    import numpy as np
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def main():
    args = parse_args()

    print("=" * 60)
    print("FID 评估")
    print("=" * 60)

    # 加载图片
    print(f"\n[加载] 真实图片: {args.real_path}")
    real_images = load_images_from_folder(args.real_path, args.img_size)
    if real_images is None:
        sys.exit(1)
    print(f"  已加载 {len(real_images)} 张真实图片")

    print(f"\n[加载] 生成图片: {args.fake_path}")
    fake_images = load_images_from_folder(args.fake_path, args.img_size)
    if fake_images is None:
        sys.exit(1)
    print(f"  已加载 {len(fake_images)} 张生成图片")

    # 初始化 InceptionV3
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n[模型] 加载 InceptionV3...")
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()  # 去掉分类头，只取特征
    inception.to(device)
    inception.eval()

    # 提取特征
    print(f"[特征] 提取真实图片特征...")
    real_feats = get_inception_features(real_images, inception, device, args.batch_size)
    print(f"[特征] 提取生成图片特征...")
    fake_feats = get_inception_features(fake_images, inception, device, args.batch_size)

    # 计算统计量
    mu_real = real_feats.mean(dim=0).numpy()
    sigma_real = torch.cov(real_feats.T).numpy()
    mu_fake = fake_feats.mean(dim=0).numpy()
    sigma_fake = torch.cov(fake_feats.T).numpy()

    # 计算 FID
    fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    print()
    print("=" * 60)
    print(f"  FID = {fid:.2f}")
    print("=" * 60)
    print()
    print("FID 分数说明:")
    print("  0-20:   非常好，生成图像与真实图像非常接近")
    print("  20-50:  良好，有可察觉的差异")
    print("  50-100: 一般，生成图像有明显质量问题")
    print("  100+:   较差，生成图像与真实图像差异很大")
    print()


if __name__ == "__main__":
    main()
