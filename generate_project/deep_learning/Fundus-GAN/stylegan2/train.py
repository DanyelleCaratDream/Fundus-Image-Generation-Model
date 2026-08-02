"""
Fundus-GAN StyleGAN2-ADA 训练脚本（统一 CLI 包装器）
=====================================================
StyleGAN2-ADA 是 NVIDIA 的 SOTA GAN 模型，其 ADA（Adaptive Discriminator Augmentation）
机制特别适合小数据集训练。

本脚本是将统一 CLI 参数转换为 StyleGAN2-ADA 原生参数的包装器。
实际训练由 stylegan2_ada/train.py 完成。

用法:
    python train.py --epochs 800 --batch_size 16 --img_size 128 \\
        --dataset_path "../../fundus/_all_images_ORIGINAL" --output_dir "./results"
"""

import argparse
import os
import sys
import subprocess
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-GAN StyleGAN2-ADA: Train StyleGAN2-ADA on fundus images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", type=int, default=200, help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.002, help="学习率（StyleGAN2 默认 0.002）")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出文件夹")

    parser.add_argument("--model_save_interval", type=int, default=50, help="模型保存间隔（对应 StyleGAN2 的 tick 间隔）")
    parser.add_argument("--image_save_interval", type=int, default=50, help="图片保存间隔")
    parser.add_argument("--preview_grid_size", type=int, default=4, help="预览网格边长")

    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="强制CPU")
    parser.add_argument("--resume", type=str, default=None, help="断点续训（StyleGAN2 .pkl 路径）")

    # StyleGAN2 特有参数
    parser.add_argument("--d_lr_factor", type=float, default=1.0, help="判别器学习率缩放")
    parser.add_argument("--d_channel_factor", type=float, default=1.0, help="判别器通道缩放")
    parser.add_argument("--freezed", type=int, default=0, help="生成器冻结层数（小数据集可用 2-4）")
    parser.add_argument("--gamma", type=float, default=2.0, help="R1 gamma（小数据集建议 0.5-2）")
    parser.add_argument("--augment", type=str, default="auto",
                        choices=["auto", "disabled", "ada"],
                        help="ADA 增强模式（auto=根据数据量自动）")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 检查 GPU
    if args.no_cuda:
        print("[WARN] StyleGAN2-ADA 需要 GPU，CPU 训练不被支持")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用，StyleGAN2-ADA 需要 CUDA")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({mem_gb:.1f} GB)")

    # 构建 StyleGAN2-ADA 原生命令
    stylegan2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stylegan2_ada")
    stylegan2_train = os.path.join(stylegan2_dir, "train_fundus.py")

    # 准备数据集（StyleGAN2 需要特定格式，用 dataset_tool.py 转换）
    # 如果数据集不是 StyleGAN2 格式，需要先转换
    dataset_path = args.dataset_path
    dataset_archive = os.path.join(args.output_dir, "dataset.zip")
    dataset_tool = os.path.join(stylegan2_dir, "dataset_tool.py")

    if not os.path.exists(dataset_archive):
        print("[PREP] 正在转换数据集为 StyleGAN2 格式...")
        cmd = [
            sys.executable, dataset_tool,
            "--source", dataset_path,
            "--dest", dataset_archive,
            "--width", str(args.img_size),
            "--height", str(args.img_size),
        ]
        subprocess.run(cmd, check=True)
        print("[PREP] 数据集转换完成")
    else:
        print("[PREP] 数据集已存在，跳过转换")

    # 构建训练命令
    cmd = [
        sys.executable, stylegan2_train,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--img_size", str(args.img_size),
        "--dataset_path", dataset_path,
        "--output_dir", args.output_dir,
        "--model_save_interval", str(args.model_save_interval),
        "--image_save_interval", str(args.image_save_interval),
        "--preview_grid_size", str(args.preview_grid_size),
        "--d_lr_factor", str(args.d_lr_factor),
        "--d_channel_factor", str(args.d_channel_factor),
        "--freezed", str(args.freezed),
        "--gamma", str(args.gamma),
        "--augment", args.augment,
    ]

    if args.resume:
        cmd.extend(["--resume", args.resume])

    print()
    print("=" * 60)
    print("正在启动 StyleGAN2-ADA 训练...")
    print("=" * 60)
    print(f"  数据集: {args.dataset_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  分辨率: {args.img_size}x{args.img_size}")
    print(f"  Epochs: {args.epochs}")
    print()

    # 执行训练
    subprocess.run(cmd, check=True)
    print("[OK] StyleGAN2-ADA 训练完成！")


if __name__ == "__main__":
    main()
