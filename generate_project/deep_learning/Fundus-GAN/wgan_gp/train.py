"""
Fundus-GAN WGAN-GP 训练脚本
============================
训练 Wasserstein GAN with Gradient Penalty 用于眼底彩照生成。
相比 DCGAN，WGAN-GP 训练更稳定，不容易模式坍塌。

用法:
    python train.py --epochs 800 --batch_size 16 --img_size 128 \\
        --dataset_path "../../../../fundus/_all_images_ORIGINAL" --output_dir "./results"
"""

import argparse
import os
import sys
import glob
import math
import time
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-GAN WGAN-GP: Train WGAN-GP on fundus images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", type=int, default=200, help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率（WGAN-GP 建议更低）")
    parser.add_argument("--g_lr", type=float, default=None, help="生成器学习率（默认同 --lr）")
    parser.add_argument("--d_lr", type=float, default=None, help="判别器学习率（默认同 --lr）")
    parser.add_argument("--b1", type=float, default=0.0, help="Adam beta1（WGAN-GP 推荐 0）")
    parser.add_argument("--b2", type=float, default=0.9, help="Adam beta2（WGAN-GP 推荐 0.9）")
    parser.add_argument("--latent_dim", type=int, default=100, help="噪声向量维度")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率（2的幂次）")
    parser.add_argument("--channels", type=int, default=3, help="图像通道数")

    # WGAN-GP 特有
    parser.add_argument("--n_critic", type=int, default=5, help="每轮 D 训练次数才训 1 次 G")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient Penalty 系数")
    parser.add_argument("--instance_noise", type=float, default=0, help="D 输入高斯噪声强度")

    parser.add_argument("--d_reduction", type=int, default=1, help="判别器通道缩减")
    parser.add_argument("--d_dropout", type=float, default=0.0, help="判别器 Dropout")

    parser.add_argument("--augment", action="store_true", default=False,
                        help="数据增强（RandomFlip + RandomRotation + ColorJitter）")

    parser.add_argument("--model_save_interval", type=int, default=50, help="模型保存间隔")
    parser.add_argument("--image_save_interval", type=int, default=50, help="图片保存间隔")
    parser.add_argument("--preview_grid_size", type=int, default=4, help="预览网格边长")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出文件夹")

    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="强制CPU")
    parser.add_argument("--resume", type=str, default=None, help="断点续训路径")

    # 文件管理
    parser.add_argument("--timestamp", action="store_true", default=False,
                        help="给输出文件夹追加时间戳，避免覆盖之前的训练结果")

    return parser.parse_args()


# ============================================================
# 环境检查
# ============================================================
def check_environment(no_cuda=False):
    print()
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    print(f"  PyTorch 版本: {torch.__version__}")
    if no_cuda:
        print("  [WARN] 用户强制使用 CPU")
        return False
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  CUDA 可用: True")
        print(f"  GPU 型号: {gpu_name}")
        print(f"  显存: {mem_gb:.1f} GB")
        _ = torch.randn(1, 1).cuda()
        return True
    else:
        print("  [WARN] CUDA 不可用，将使用 CPU")
        return False


# ============================================================
# 数据集
# ============================================================
class FundusDataset(Dataset):
    """眼底彩照数据集，预加载到内存。"""

    def __init__(self, root, aug_transform=None, fixed_transform=None):
        self.aug_transform = aug_transform
        self.images = []

        if not os.path.isdir(root):
            raise FileNotFoundError(f"数据集路径不存在: {root}")

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(root, ext)))
        image_paths = sorted(list(set(image_paths)))

        if len(image_paths) == 0:
            raise ValueError(f"在 {root} 中没有找到任何图片文件！")

        print(f"  找到 {len(image_paths)} 张图片，正在预加载到内存...")
        t0 = time.time()
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            if fixed_transform:
                img = fixed_transform(img)
            self.images.append(img)
        elapsed = time.time() - t0
        print(f"  预加载完成，耗时 {elapsed:.1f}s，"
              f"内存约 {len(self.images) * 128 * 128 * 3 * 4 / 1024 / 1024:.0f} MB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.aug_transform:
            img = self.aug_transform(img)
        return img, 0


# ============================================================
# 模型定义
# ============================================================
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """生成器：与 DCGAN 版本相同。Upsample + Conv 避免棋盘格。"""

    def __init__(self, img_size=128, latent_dim=100, channels=3):
        super(Generator, self).__init__()
        if not is_power_of_two(img_size) or img_size < 16:
            raise ValueError(f"img_size 必须是 2 的幂次，当前: {img_size}")

        self.init_size = 4
        n_upsample = int(math.log2(img_size // self.init_size))
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size * self.init_size))

        conv_blocks = []
        in_ch = 512
        for i in range(n_upsample):
            is_last = (i == n_upsample - 1)
            out_ch = channels if is_last else in_ch // 2
            conv_blocks.append(nn.Upsample(scale_factor=2))
            conv_blocks.append(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1))
            if not is_last:
                conv_blocks.append(nn.BatchNorm2d(out_ch))
                conv_blocks.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                conv_blocks.append(nn.Tanh())
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Critic(nn.Module):
    """WGAN Critic（判别器）：输出不加 Sigmoid，输出标量分数。"""

    def __init__(self, img_size=128, channels=3, reduction=1, dropout=0.0):
        super(Critic, self).__init__()
        if not is_power_of_two(img_size) or img_size < 16:
            raise ValueError(f"img_size 必须是 2 的幂次，当前: {img_size}")

        n_downsample = int(math.log2(img_size // 4))
        layers = []
        in_ch = channels
        for i in range(n_downsample):
            out_ch = max(4, (64 * (2**i)) // reduction)
            layers.append(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i > 0:
                if dropout > 0:
                    layers.append(nn.Dropout2d(dropout))
                layers.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch
        self.model = nn.Sequential(*layers)
        ds_size = img_size // (2**n_downsample)
        self.fc = nn.Linear(out_ch * ds_size * ds_size, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.fc(out)


# ============================================================
# Gradient Penalty
# ============================================================
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算 WGAN-GP 的梯度惩罚项。"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(batch_size, 1, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ============================================================
# 辅助函数
# ============================================================
def save_loss_plot(g_losses, d_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss", alpha=0.8)
    plt.plot(d_losses, label="Critic Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_banner(args, device, dataset_len):
    print()
    print("=" * 60)
    print("Fundus-WGAN-GP 训练开始")
    print("=" * 60)
    print(f"  数据集路径:    {args.dataset_path}")
    print(f"  图片数量:      {dataset_len}")
    print(f"  输出目录:      {args.output_dir}")
    print(f"  设备:          {device}")
    print(f"  图像分辨率:    {args.img_size}x{args.img_size}")
    print(f"  批次大小:      {args.batch_size}")
    print(f"  训练轮数:      {args.epochs}")
    print(f"  学习率:        {args.lr}")
    print(f"  n_critic:      {args.n_critic}")
    print(f"  lambda_gp:     {args.lambda_gp}")
    print(f"  模型保存间隔:  每 {args.model_save_interval} 轮")
    print(f"  图片保存间隔:  每 {args.image_save_interval} 轮")
    print(f"  预览网格:      {args.preview_grid_size}x{args.preview_grid_size}")
    print()


# ============================================================
# 主训练流程
# ============================================================
def main():
    args = parse_args()

    # 时间戳：新训练自动追加，resume续训不加
    if not args.resume:
        from datetime import datetime
        ts = datetime.now().strftime("%d%m%y_%H%M%S")
        args.output_dir = args.output_dir.rstrip("/\\") + f"_{ts}"

    if not is_power_of_two(args.img_size) or args.img_size < 16:
        print(f"[ERROR] img_size={args.img_size} 不是 2 的幂次")
        sys.exit(1)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    models_dir = os.path.join(args.output_dir, "models")
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    use_cuda = check_environment(args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # 数据增强（随机部分：每轮重新应用）
    aug_transform = None
    if args.augment:
        print("  [AUG] 数据增强: RandomFlip + RandomRotation + ColorJitter")
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ])

    # 固定变换（预加载时一次性做完）
    fixed_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * args.channels, [0.5] * args.channels),
    ])

    dataset = FundusDataset(args.dataset_path, aug_transform=aug_transform, fixed_transform=fixed_transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0,
        pin_memory=use_cuda,
    )

    print_banner(args, device, len(dataset))

    generator = Generator(img_size=args.img_size, latent_dim=args.latent_dim, channels=args.channels)
    critic = Critic(img_size=args.img_size, channels=args.channels,
                    reduction=args.d_reduction, dropout=args.d_dropout)

    generator.to(device)
    critic.to(device)
    generator.apply(weights_init_normal)
    critic.apply(weights_init_normal)

    g_lr = args.g_lr if args.g_lr is not None else args.lr
    d_lr = args.d_lr if args.d_lr is not None else args.lr
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(critic.parameters(), lr=d_lr, betas=(args.b1, args.b2))

    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载 checkpoint: {args.resume}")
            cp = torch.load(args.resume, map_location=device)
            generator.load_state_dict(cp["generator_state_dict"])
            critic.load_state_dict(cp["critic_state_dict"])
            optimizer_G.load_state_dict(cp["optimizer_G_state_dict"])
            optimizer_D.load_state_dict(cp["optimizer_D_state_dict"])
            start_epoch = cp.get("epoch", -1) + 1
            print(f"  从 epoch {start_epoch} 继续训练")
        else:
            print("  checkpoint 不存在，从头开始训练")

    fixed_z = torch.randn(args.preview_grid_size**2, args.latent_dim, device=device)

    log_file = os.path.join(logs_dir, "training.log")
    log_mode = "a" if args.resume else "w"
    with open(log_file, log_mode, encoding="utf-8") as f:
        if log_mode == "w":
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Args: {json.dumps(vars(args))}\n\n")
        else:
            f.write(f"\n--- Resumed at epoch {start_epoch} ---\n")

    g_losses, d_losses = [], []
    start_time = time.time()
    print("提示: 按 Ctrl+C 可安全中断训练\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            g_updates = 0
            d_updates = 0

            noise_std = args.instance_noise * max(0, 1 - epoch / args.epochs) if args.instance_noise > 0 else 0

            for i, (real_imgs, _) in enumerate(dataloader):
                batch = real_imgs.shape[0]
                real_imgs = real_imgs.to(device)
                real_input = real_imgs
                if noise_std > 0:
                    real_input = real_imgs + torch.randn_like(real_imgs) * noise_std

                # Train Critic
                optimizer_D.zero_grad()
                z = torch.randn(batch, args.latent_dim, device=device)
                fake_imgs = generator(z)

                real_validity = critic(real_input)
                fake_input = fake_imgs.detach()
                if noise_std > 0:
                    fake_input = fake_imgs.detach() + torch.randn_like(fake_imgs) * noise_std
                fake_validity = critic(fake_input)
                gp = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data, device)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gp
                d_loss.backward()
                optimizer_D.step()
                epoch_d_loss += d_loss.item()
                d_updates += 1

                # Train Generator (每 n_critic 次 D 更新后训一次)
                if i % args.n_critic == 0:
                    optimizer_G.zero_grad()
                    z = torch.randn(batch, args.latent_dim, device=device)
                    gen_imgs = generator(z)
                    g_loss = -torch.mean(critic(gen_imgs))
                    g_loss.backward()
                    optimizer_G.step()
                    epoch_g_loss += g_loss.item()
                    g_updates += 1

                if i % 10 == 0 or i == len(dataloader) - 1:
                    print(f"  [Epoch {epoch:04d}/{args.epochs}] "
                          f"[Batch {i:04d}/{len(dataloader)}] "
                          f"[D: {d_loss.item():.4f}] [G: {g_loss.item():.4f}]")

            avg_g = epoch_g_loss / max(g_updates, 1)
            avg_d = epoch_d_loss / max(d_updates, 1)
            g_losses.append(avg_g)
            d_losses.append(avg_d)
            epoch_time = time.time() - epoch_start

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch}: D={avg_d:.4f}, G={avg_g:.4f}, time={epoch_time:.1f}s\n")

            # 保存模型
            if epoch % args.model_save_interval == 0 or epoch == args.epochs:
                torch.save({
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "args": args,
                }, os.path.join(models_dir, f"checkpoint_epoch_{epoch:06d}.pth"))
                print(f"  [SAVE] 模型已保存: epoch {epoch}")

            # 保存图片
            if epoch % args.image_save_interval == 0 or epoch == 1 or epoch == args.epochs:
                generator.eval()
                with torch.no_grad():
                    fixed_gen = generator(fixed_z)
                # tanh [-1,1] → [0,1] 手动映射，保证颜色准确
                save_image((fixed_gen + 1) / 2, os.path.join(images_dir, f"epoch_{epoch:04d}.png"),
                           nrow=args.preview_grid_size, normalize=False)
                generator.train()
                print(f"  [IMG] 预览图已保存: epoch {epoch}")

            if epoch % max(args.image_save_interval // 2, 1) == 0:
                save_loss_plot(g_losses, d_losses, os.path.join(logs_dir, "loss_curve.png"))

    except KeyboardInterrupt:
        print("\n\n[WARN] 训练被中断")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
        }, os.path.join(models_dir, "checkpoint_INTERRUPTED.pth"))
        print("  [SAVE] 已保存中断时的模型")

    save_loss_plot(g_losses, d_losses, os.path.join(logs_dir, "loss_curve_final.png"))
    print("\n[OK] 训练结束！")
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
