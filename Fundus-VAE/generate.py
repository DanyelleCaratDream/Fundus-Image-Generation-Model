"""
Fundus-VAE 生成脚本
====================
用训练好的 VAE 模型批量生成眼底彩照。

用法:
    python generate.py --checkpoint "./results/models/final_model.pth" \\
        --num_images 100 --img_size 128 --output_dir "./results/generated"
"""

import argparse
import os
import sys

import torch
import torchvision
from torchvision.utils import save_image

from train import VAE, check_environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-VAE: Generate fundus images from trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径 (.pth)")
    parser.add_argument("--num_images", type=int, default=100, help="生成图片数量")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率，须与训练时一致")
    parser.add_argument("--latent_dim", type=int, default=128, help="隐空间维度，须与训练时一致")
    parser.add_argument("--dim", type=int, default=32, help="模型容量参数，须与训练时一致")
    parser.add_argument("--output_dir", type=str, default="./generated", help="输出文件夹")
    parser.add_argument("--grid_size", type=int, default=10, help="网格大小 (如10=10x10, 0=保存单张)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="强制使用CPU")
    parser.add_argument("--timestamp", action="store_true", default=False,
                        help="给输出文件夹追加时间戳，避免覆盖之前结果")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.timestamp:
        from datetime import datetime
        ts = datetime.now().strftime("%d%m%y_%H%M%S")
        args.output_dir = args.output_dir.rstrip("/\\") + f"_{ts}"

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] 模型文件不存在: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 环境检查
    use_cuda = check_environment(args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # 加载模型
    model = VAE(
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        channels=3,
        dim=args.dim,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 兼容旧格式（直接保存的 state_dict）
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"[OK] 已加载模型: {args.checkpoint}")

    # 生成图片
    print(f"[GEN] 正在生成 {args.num_images} 张图片...")

    batch_size = min(args.num_images, 64)
    all_images = []

    with torch.no_grad():
        remaining = args.num_images
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, args.latent_dim, device=device)
            imgs = model.generate(z)
            all_images.append(imgs.cpu())
            remaining -= bs

    all_images = torch.cat(all_images, dim=0)
    print(f"[OK] 已生成 {len(all_images)} 张图片")

    # 保存网格图
    if args.grid_size > 0:
        grid_path = os.path.join(args.output_dir, f"grid_{args.num_images}.png")
        save_image(all_images, grid_path, nrow=args.grid_size, normalize=True)
        print(f"[IMG] 网格图已保存: {grid_path}")

    # 保存单张图片
    singles_dir = os.path.join(args.output_dir, "singles")
    os.makedirs(singles_dir, exist_ok=True)
    for idx in range(min(args.num_images, len(all_images))):
        save_image(
            all_images[idx],
            os.path.join(singles_dir, f"sample_{idx:04d}.png"),
            normalize=True,
        )
    print(f"[IMG] 单张图已保存到: {singles_dir}")
    print()
    print("[OK] 生成完成！")


if __name__ == "__main__":
    main()
