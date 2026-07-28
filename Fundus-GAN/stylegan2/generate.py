"""
Fundus-GAN StyleGAN2-ADA 生成脚本（统一 CLI 包装器）
=====================================================
用训练好的 StyleGAN2-ADA 模型生成眼底彩照。

用法:
    python generate.py --checkpoint "./results/00000-xxx/network-snapshot-000200.pkl" \\
        --num_images 100 --img_size 128 --output_dir "./generated"
"""

import argparse
import os
import sys
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-GAN StyleGAN2-ADA: Generate fundus images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型路径 (.pkl)")
    parser.add_argument("--num_images", type=int, default=100, help="生成图片数量")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率")
    parser.add_argument("--output_dir", type=str, default="./generated", help="输出文件夹")
    parser.add_argument("--grid_size", type=int, default=10, help="网格大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--trunc", type=float, default=0.8, help="截断系数 (越低越真实但多样性降低)")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stylegan2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stylegan2_ada")
    generate_script = os.path.join(stylegan2_dir, "generate_fundus.py")

    cmd = [
        sys.executable, generate_script,
        "--network", args.checkpoint,
        "--outdir", args.output_dir,
        "--count", str(args.num_images),
        "--grid_size", str(args.grid_size),
        "--trunc", str(args.trunc),
        "--seed_start", str(args.seed),
    ]

    print(f"正在启动 StyleGAN2-ADA 生成...")
    print(f"  模型: {args.checkpoint}")
    print(f"  数量: {args.num_images}")
    print(f"  输出: {args.output_dir}")

    subprocess.run(cmd, check=True)
    print("[OK] 生成完成！")


if __name__ == "__main__":
    main()
