"""
Fundus-GAN DCGAN 生成脚本
=========================
用训练好的 DCGAN 生成器批量生成眼底彩照。

用法:
    python generate.py --checkpoint "./results/models/checkpoint_epoch_000800.pth" \\
        --num_images 100 --img_size 128 --output_dir "./generated"
"""

import argparse
import os
import sys

import torch
from torchvision.utils import save_image

from train import Generator, check_environment, is_power_of_two


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-GAN DCGAN: Generate fundus images from trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径 (.pth)")
    parser.add_argument("--num_images", type=int, default=100, help="生成图片数量")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率，须与训练时一致")
    parser.add_argument("--latent_dim", type=int, default=100, help="噪声维度，须与训练时一致")
    parser.add_argument("--channels", type=int, default=3, help="图像通道数")
    parser.add_argument("--output_dir", type=str, default="./generated", help="输出文件夹")
    parser.add_argument("--grid_size", type=int, default=10, help="网格大小 (0=保存单张)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="强制使用CPU")
    parser.add_argument("--timestamp", action="store_true", default=False,
                        help="给输出文件夹追加时间戳，避免覆盖之前的结果")
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

    use_cuda = check_environment(args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # 加载模型
    generator = Generator(
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        channels=args.channels,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
    else:
        generator.load_state_dict(checkpoint)
    generator.to(device)
    generator.eval()
    print(f"[OK] 已加载模型: {args.checkpoint}")

    # 生成
    print(f"[GEN] 正在生成 {args.num_images} 张图片...")
    batch_size = min(args.num_images, 64)
    all_images = []

    with torch.no_grad():
        remaining = args.num_images
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, args.latent_dim, device=device)
            imgs = generator(z)
            all_images.append(imgs.cpu())
            remaining -= bs

    all_images = torch.cat(all_images, dim=0)
    print(f"[OK] 已生成 {len(all_images)} 张图片")

    # tanh [-1,1] → [0,1] 手动映射后再保存，保证颜色准确
    all_images = (all_images + 1) / 2

    # 网格图
    if args.grid_size > 0:
        grid_path = os.path.join(args.output_dir, f"grid_{args.num_images}.png")
        save_image(all_images, grid_path, nrow=args.grid_size, normalize=False)
        print(f"[IMG] 网格图已保存: {grid_path}")

    # 单张保存
    singles_dir = os.path.join(args.output_dir, "singles")
    os.makedirs(singles_dir, exist_ok=True)
    for idx in range(min(args.num_images, len(all_images))):
        save_image(
            all_images[idx],
            os.path.join(singles_dir, f"sample_{idx:04d}.png"),
            normalize=False,
        )
    print(f"[IMG] 单张图已保存到: {singles_dir}")
    print("[OK] 生成完成！")


if __name__ == "__main__":
    main()
