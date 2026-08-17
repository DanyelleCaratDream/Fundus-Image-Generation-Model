"""
Fundus-VAE 训练脚本
====================
训练变分自编码器 (VAE) 用于眼底彩照生成。
支持 vanilla VAE 和 beta-VAE（通过 --beta 参数调节）。

用法:
    python train.py --epochs 500 --batch_size 32 --img_size 128 \\
        --dataset_path "../../../fundus/_all_images_ORIGINAL" --output_dir "./results"
"""

import argparse
import os
import sys
import time
import json
import math
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-VAE: Train Variational Autoencoder on fundus images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 训练超参
    parser.add_argument("--epochs", type=int, default=200, help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率（必须是 4 的 2 的幂次倍）")
    parser.add_argument("--latent_dim", type=int, default=128, help="隐空间维度")
    parser.add_argument("--beta", type=float, default=1.0, help="KL 权重 (1.0=标准VAE, >1=beta-VAE)")
    parser.add_argument("--dim", type=int, default=32, help="基础通道数（控制模型容量）")

    # 保存与输出
    parser.add_argument("--model_save_interval", type=int, default=50, help="每隔多少轮保存一次模型")
    parser.add_argument("--image_save_interval", type=int, default=50, help="每隔多少轮保存一次生成图片")
    parser.add_argument("--preview_grid_size", type=int, default=4, help="预览图网格边长（如4表示4x4=16张）")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出文件夹路径")

    # 数据与硬件
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集文件夹路径（里面直接放图片）")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="强制使用CPU")
    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")
    parser.add_argument("--precache", action="store_true", default=True, help="预加载图片到内存加速")
    parser.add_argument("--no_precache", dest="precache", action="store_false")
    parser.add_argument("--amp", action="store_true", default=False, help="启用混合精度训练")

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

    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA 可用: {'是' if cuda_avail else '否'}")

    if cuda_avail:
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU 型号: {gpu_name}")
        print(f"  显存: {mem_gb:.1f} GB")
        # 快速 GPU 测试
        try:
            t = torch.tensor([1.0, 2.0]).cuda()
            torch.cuda.synchronize()
            del t
            print("  GPU 测试: 通过")
        except Exception as e:
            print(f"  GPU 测试: 失败 - {e}")
            return False
    else:
        print("  [WARN] 使用 CPU 训练将非常缓慢！")

    print("=" * 60)
    print()
    return cuda_avail


# ============================================================
# 数据集
# ============================================================
class FundusDataset(Dataset):
    """直接从文件夹读取眼底图，支持预加载到内存。"""

    VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, root, transform=None, precache=False):
        self.root = root
        self.transform = transform

        # 扫描文件夹下所有图片
        self.image_paths = []
        for ext in self.VALID_EXT:
            pattern = os.path.join(root, "*" + ext)
            import glob
            self.image_paths.extend(glob.glob(pattern))
            pattern = os.path.join(root, "*" + ext.upper())
            self.image_paths.extend(glob.glob(pattern))

        self.image_paths = sorted(list(set(self.image_paths)))

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {root} 中没有找到任何图片文件！")

        print(f"  找到 {len(self.image_paths)} 张图片")

        # 预加载到内存
        self.cached_data = None
        if precache:
            self._precache()

    def _precache(self):
        print("  正在预加载图片到内存...", end=" ", flush=True)
        t0 = time.time()
        tensors = []
        for p in self.image_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            tensors.append(img)
        self.cached_data = torch.stack(tensors)
        mem_mb = self.cached_data.numel() * self.cached_data.element_size() / (1024 * 1024)
        print(f"完成 ({len(tensors)} 张, {mem_mb:.1f} MB, {time.time()-t0:.1f}s)")

    def __len__(self):
        if self.cached_data is not None:
            return len(self.cached_data)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cached_data is not None:
            return self.cached_data[idx], 0
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ============================================================
# VAE 模型
# ============================================================
class Encoder(nn.Module):
    """卷积编码器：图像 -> mu, logvar"""

    def __init__(self, latent_dim=128, img_size=128, channels=3, dim=32):
        super().__init__()
        n_layers = int(math.log2(img_size // 4))

        in_c = channels
        blocks = []
        for i in range(n_layers):
            out_c = min(dim * (2**i), dim * 8)
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)

        final_size = img_size // (2**n_layers)  # should be 4
        final_c = in_c * final_size * final_size
        self.fc_mu = nn.Linear(final_c, latent_dim)
        self.fc_logvar = nn.Linear(final_c, latent_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """卷积解码器：z -> 图像"""

    def __init__(self, latent_dim=128, img_size=128, channels=3, dim=32):
        super().__init__()
        self.img_size = img_size

        n_layers = int(math.log2(img_size // 4))
        start_channels = dim * 8

        self.init_linear = nn.Linear(latent_dim, start_channels * 4 * 4)

        in_c = start_channels
        blocks = []
        for i in range(n_layers):
            out_c = max(dim, start_channels // (2 ** (i + 1)))
            blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            ))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_c, channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.init_linear(z)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.blocks(x)
        x = self.final(x)
        if x.size(2) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        return x


class VAE(nn.Module):
    """完整的 VAE 模型（编码器 + 解码器）"""

    def __init__(self, latent_dim=128, img_size=128, channels=3, dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, img_size, channels, dim)
        self.decoder = Decoder(latent_dim, img_size, channels, dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + std * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def generate(self, z):
        """从隐向量生成图像"""
        return self.decoder(z)

    def encode(self, x):
        """将图像编码为隐向量（返回 mu）"""
        mu, _ = self.encoder(x)
        return mu


# ============================================================
# VAE 损失函数
# ============================================================
def vae_loss(recon, target, mu, logvar, beta=1.0):
    """
    VAE 损失 = 重建损失 (MSE) + beta * KL 散度

    参数:
        recon: 重建图像 [B, C, H, W] (tanh 输出, 范围 [-1, 1])
        target: 原始图像 [B, C, H, W]
        mu: 隐空间均值 [B, latent_dim]
        logvar: 隐空间对数方差 [B, latent_dim]
        beta: KL 权重 (1.0 = 标准 VAE, >1 = beta-VAE)
    """
    # 重建损失：MSE
    recon_loss = F.mse_loss(recon, target, reduction="sum") / recon.size(0)

    # KL 散度：KL(N(mu, std) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ============================================================
# 辅助函数
# ============================================================
def save_image_grid(tensor, filepath, nrow=8):
    """保存多张图片到一张网格图中"""
    torchvision.utils.save_image(
        tensor, filepath, nrow=nrow, normalize=True, padding=2,
    )


def save_loss_curve(train_losses, recon_losses, kl_losses, save_path):
    """绘制并保存 Loss 曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Total Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, label="Reconstruction Loss", alpha=0.8, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(kl_losses, label="KL Loss", alpha=0.8, color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("KL Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_banner(args, device, dataset_len):
    print()
    print("=" * 60)
    print("Fundus-VAE 训练开始")
    print("=" * 60)
    print(f"  数据集路径:    {args.dataset_path}")
    print(f"  图片数量:      {dataset_len}")
    print(f"  输出目录:      {args.output_dir}")
    print(f"  设备:          {device}")
    print(f"  图像分辨率:    {args.img_size}x{args.img_size}")
    print(f"  批次大小:      {args.batch_size}")
    print(f"  训练轮数:      {args.epochs}")
    print(f"  学习率:        {args.lr}")
    print(f"  隐空间维度:    {args.latent_dim}")
    print(f"  Beta (KL权重): {args.beta}")
    print(f"  模型容量(dim): {args.dim}")
    print(f"  模型保存间隔:  每 {args.model_save_interval} 轮")
    print(f"  图片保存间隔:  每 {args.image_save_interval} 轮")
    print(f"  预览网格:      {args.preview_grid_size}x{args.preview_grid_size}")
    print(f"  混合精度:      {'启用' if args.amp else '关闭'}")
    if args.beta > 1:
        print(f"  [beta-VAE] beta={args.beta} > 1, 启用更强的隐空间解耦")
    print("=" * 60)
    print()


# ============================================================
# 主训练流程
# ============================================================
def main():
    args = parse_args()

    # 时间戳：自动追加到输出目录，避免覆盖之前结果
    if args.timestamp:
        ts = datetime.now().strftime("%d%m%y_%H%M%S")
        args.output_dir = args.output_dir.rstrip("/\\") + f"_{ts}"

    # 验证 img_size 合法性
    size = args.img_size
    n_layers = int(math.log2(size // 4))
    if 4 * (2**n_layers) != size:
        print(f"[ERROR] img_size ({size}) 必须是 4 * power-of-2 (如 64, 128, 256)")
        sys.exit(1)

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 环境检查
    use_cuda = check_environment(args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    models_dir = os.path.join(args.output_dir, "models")
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 数据集
    dataset = FundusDataset(args.dataset_path, transform=transform, precache=args.precache)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if args.precache else min(args.num_workers, 4),
        pin_memory=use_cuda, drop_last=True,
    )

    print_banner(args, device, len(dataset))

    # 初始化模型
    model = VAE(
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        channels=3,
        dim=args.dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 混合精度
    scaler = torch.amp.GradScaler("cuda") if args.amp and use_cuda else None

    # 固定噪声用于生成预览
    n_preview = args.preview_grid_size ** 2
    fixed_noise = torch.randn(n_preview, args.latent_dim, device=device)

    # 固定真实图像用于重建对比
    fixed_real = None

    # 断点续训
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载 checkpoint: {args.resume}")
            cp = torch.load(args.resume, map_location=device)
            model.load_state_dict(cp["model_state_dict"])
            optimizer.load_state_dict(cp["optimizer_state_dict"])
            if scaler and "scaler_state_dict" in cp:
                scaler.load_state_dict(cp["scaler_state_dict"])
            start_epoch = cp.get("epoch", -1) + 1
            best_loss = cp.get("best_loss", float("inf"))
            print(f"  从 epoch {start_epoch} 继续训练")
        else:
            print(f"  checkpoint 不存在 ({args.resume})，从头开始训练")

    # 生成初始随机样本验证输出正常
    with torch.no_grad():
        initial_gen = model.generate(torch.randn(n_preview, args.latent_dim, device=device))
        save_image_grid(initial_gen,
                        os.path.join(images_dir, "initial_random.png"),
                        nrow=args.preview_grid_size)
    print(f"  初始样本已保存: {images_dir}")
    print()

    # 训练日志
    log_file = os.path.join(logs_dir, "training.log")
    log_mode = "a" if args.resume else "w"
    with open(log_file, log_mode, encoding="utf-8") as f:
        if log_mode == "w":
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Args: {json.dumps(vars(args))}\n\n")
        else:
            f.write(f"\n--- Resumed at epoch {start_epoch} ---\n")

    # ========== 训练循环 ==========
    train_losses = []
    recon_losses = []
    kl_losses = []
    train_start_time = time.time()

    print("=" * 60)
    print("训练开始")
    print("=" * 60)
    print("提示: 按 Ctrl+C 可安全中断训练\n")

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            epoch_losses = []
            epoch_recon = []
            epoch_kl = []

            model.train()

            for real_data, _ in dataloader:
                real_data = real_data.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with torch.amp.autocast("cuda"):
                        recon, mu, logvar = model(real_data)
                        loss, rl, kl = vae_loss(
                            recon.float(), real_data, mu, logvar, beta=args.beta,
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon, mu, logvar = model(real_data)
                    loss, rl, kl = vae_loss(
                        recon, real_data, mu, logvar, beta=args.beta,
                    )
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
                epoch_recon.append(rl.item())
                epoch_kl.append(kl.item())

            # 记录 epoch 统计
            avg_loss = np.mean(epoch_losses)
            avg_recon = np.mean(epoch_recon)
            avg_kl = np.mean(epoch_kl)
            best_loss = min(best_loss, avg_loss)
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - train_start_time

            # 打印进度
            epoch_str = f"[{epoch+1:>{len(str(args.epochs))}}/{args.epochs}]"
            print(f"  {epoch_str}  loss={avg_loss:.1f}  recon={avg_recon:.1f}  "
                  f"KL={avg_kl:.1f}  [{epoch_time:.0f}s]")

            # 写入日志
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch+1}: loss={avg_loss:.1f}, recon={avg_recon:.1f}, "
                       f"kl={avg_kl:.1f}, time={epoch_time:.1f}s\n")

            # 保存模型
            if (epoch + 1) % args.model_save_interval == 0 or epoch == args.epochs - 1:
                cp = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "args": args,
                }
                if scaler:
                    cp["scaler_state_dict"] = scaler.state_dict()
                save_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch+1:06d}.pth")
                torch.save(cp, save_path)
                print(f"  [SAVE] 模型已保存: {save_path}")

            # 保存生成样本
            if (epoch + 1) % args.image_save_interval == 0 or epoch == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    # 生成样本
                    gen = model.generate(fixed_noise)
                    save_image_grid(
                        gen,
                        os.path.join(images_dir, f"generated_epoch_{epoch+1:06d}.png"),
                        nrow=args.preview_grid_size,
                    )

                    # 重建样本
                    if fixed_real is None:
                        fixed_real = real_data[:n_preview].clone()

                    recon, _, _ = model(fixed_real)
                    both = torch.cat([fixed_real, recon], dim=0)
                    save_image_grid(
                        both,
                        os.path.join(images_dir, f"recon_epoch_{epoch+1:06d}.png"),
                        nrow=args.preview_grid_size,
                    )
                    print(f"  [IMG] 生成图和重建图已保存 (epoch {epoch+1})")

                model.train()

            # 保存 Loss 曲线
            if (epoch + 1) % max(args.image_save_interval // 2, 1) == 0:
                save_loss_curve(
                    train_losses, recon_losses, kl_losses,
                    os.path.join(logs_dir, "loss_curve.png"),
                )

    except KeyboardInterrupt:
        print("\n\n[WARN] 训练被用户中断 (Ctrl+C)")
        interrupt_path = os.path.join(models_dir, "checkpoint_INTERRUPTED.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "args": args,
        }, interrupt_path)
        print(f"  [SAVE] 已保存中断时的模型: {interrupt_path}")

    # 训练结束
    total_time = time.time() - train_start_time
    print()
    print("=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  总轮数: {args.epochs}")
    print(f"  最优 Loss: {best_loss:.1f}")
    print()

    # 保存最终模型
    final_path = os.path.join(models_dir, "final_model.pth")
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "best_loss": best_loss,
        "args": args,
    }, final_path)
    print(f"  最终模型 -> {final_path}")

    # 生成最终样本和重建图
    model.eval()
    with torch.no_grad():
        gen = model.generate(torch.randn(64, args.latent_dim, device=device))
        save_image_grid(gen, os.path.join(args.output_dir, "final_generated.png"), nrow=8)

        if fixed_real is not None:
            recon, _, _ = model(fixed_real)
            both = torch.cat([fixed_real, recon], dim=0)
            save_image_grid(both, os.path.join(args.output_dir, "final_reconstructions.png"), nrow=8)
            print(f"  最终重建图 -> {os.path.join(args.output_dir, 'final_reconstructions.png')}")

    # 最终 Loss 曲线
    save_loss_curve(
        train_losses, recon_losses, kl_losses,
        os.path.join(logs_dir, "loss_curve_final.png"),
    )

    print()
    print("输出目录:")
    print(f"  {os.path.abspath(args.output_dir)}")
    print("  - images/   : 训练过程生成的预览图")
    print("  - models/   : 模型权重文件")
    print("  - logs/     : 训练日志和 Loss 曲线")
    print()


if __name__ == "__main__":
    main()
