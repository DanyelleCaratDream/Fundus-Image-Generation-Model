"""
Fundus-Diffusion DDPM 训练脚本
===============================
训练 Denoising Diffusion Probabilistic Models 用于眼底彩照生成。
DDPM 是目前生成质量最高的模型之一，训练稳定，不会模式坍塌。

理论简介:
- 前向过程: 逐步对图像加高斯噪声，直到完全变成随机噪声
- 反向过程: 训练一个 UNet 预测噪声，从随机噪声逐步去噪还原图像
- 损失函数: 预测噪声与真实噪声的 MSE

用法:
    python train.py --epochs 500 --batch_size 16 --img_size 128 \\
        --dataset_path "../../../../fundus/_all_images_ORIGINAL" --output_dir "./results"

采样（训练后）:
    python generate.py --checkpoint "./results/models/checkpoint_epoch_000500.pth" \\
        --num_images 64 --output_dir "./results/generated"
"""

import argparse
import os
import sys
import math
import time
import json
from datetime import datetime

import numpy as np
from PIL import Image
from PIL import ImageFilter  # 用于条件图预处理
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-Diffusion DDPM: Train DDPM on fundus images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", type=int, default=500, help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率")
    parser.add_argument("--channels", type=int, default=3, help="图像通道数")

    # Diffusion 特有参数
    parser.add_argument("--timesteps", type=int, default=1000, help="扩散步数（训练用）")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="噪声调度起始值")
    parser.add_argument("--beta_end", type=float, default=0.02, help="噪声调度结束值")
    parser.add_argument("--base_dim", type=int, default=128, help="UNet 基础通道数")
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="UNet 每层通道数倍数")
    parser.add_argument("--attn_layers", type=int, nargs="+", default=[2],
                        help="在哪些下采样层添加自注意力 (从0开始)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA 衰减率（0.9999 表示保留 99.99% 历史权重，默认 0.9999）")
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True,
                        help="使用 FiLM 调制（scale_shift_norm），改善细节保留（推荐）")
    parser.add_argument("--no_scale_shift_norm", action="store_true",
                        help="禁用 FiLM 调制，使用旧版加性偏置（兼容旧 checkpoint）")

    # 保存与输出
    parser.add_argument("--model_save_interval", type=int, default=50, help="每隔多少轮保存一次模型")
    parser.add_argument("--image_save_interval", type=int, default=50, help="每隔多少轮保存一次生成图")
    parser.add_argument("--preview_grid_size", type=int, default=4, help="预览图网格边长")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出文件夹")

    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="强制CPU")
    parser.add_argument("--resume", type=str, default=None, help="断点续训 checkpoint 路径")
    parser.add_argument("--color_weight", type=float, default=0.01,
                        help="颜色正则化权重（越大颜色约束越强，建议 0.005~0.05）")
    parser.add_argument("--use_lpips", action="store_true", default=False,
                        help="启用 LPIPS 感知损失（需要 pip install lpips）")
    parser.add_argument("--lpips_weight", type=float, default=0.1,
                        help="LPIPS 感知损失权重（默认 0.1，仅在 --use_lpips 时生效）")
    parser.add_argument("--cond_path", type=str, default=None,
                        help="条件图路径（血管骨架等），开启条件扩散训练")

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
        print("  [WARN] CUDA 不可用，将使用 CPU（非常慢）")
        return False


# ============================================================
# 数据集
# ============================================================
class FundusDataset(Dataset):
    """直接从文件夹读取眼底图，可选加载条件图（血管骨架）。"""

    def __init__(self, root, transform=None, cond_root=None, cond_transform=None, img_size=128):
        self.transform = transform
        self.cond_root = cond_root
        self.cond_transform = cond_transform or transform
        self.img_size = img_size
        self.image_paths = []
        if not os.path.isdir(root):
            raise FileNotFoundError(f"数据集路径不存在: {root}")

        import glob
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(root, ext)))
        self.image_paths = sorted(list(set(self.image_paths)))

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {root} 中没有找到任何图片！")
        print(f"  找到 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 条件图（血管骨架）
        cond = None
        if self.cond_root:
            basename = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
            cond_path = os.path.join(self.cond_root, basename + ".png")
            if os.path.isfile(cond_path):
                if _HAS_CV2:
                    cond_cv = cv2.imread(cond_path, cv2.IMREAD_GRAYSCALE)
                    if cond_cv is not None:
                        cond_cv = cv2.resize(cond_cv, (self.img_size, self.img_size))
                        cond = torch.from_numpy(cond_cv).float() / 255.0 * 2.0 - 1.0
                        cond = cond.unsqueeze(0)  # [1, H, W]
                else:
                    cond_pil = Image.open(cond_path).convert("L").resize((self.img_size, self.img_size))
                    cond = (torch.from_numpy(np.array(cond_pil)).float() / 255.0 * 2.0 - 1.0).unsqueeze(0)
            if cond is None:
                cond = torch.zeros(1, self.img_size, self.img_size)
        return img, cond, idx


# ============================================================
# 扩散过程工具函数
# ============================================================
def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦噪声调度（cosine schedule），相比线性调度效果更好。
    参考: "Improved DDPM" 论文
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, max=0.999)


def extract(v, t, x_shape):
    """从预计算系数中提取 t 时刻的值，reshape 用于广播。"""
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusion:
    """
    高斯扩散过程管理类。
    处理前向加噪、后向采样的所有数学计算。
    支持 DDPM 和 DDIM 两种采样方式。
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule="cosine"):
        self.timesteps = timesteps

        # 选择噪声调度
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, timesteps)

        # 预计算所有扩散相关的系数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # 前向过程系数
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 后向过程系数
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def to(self, device):
        """将所有系数移至指定设备"""
        for attr in ['betas', 'alphas', 'alphas_cumprod',
                      'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                      'posterior_variance']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        """
        前向加噪: x_t = sqrt(alpha_cumprod[t]) * x_0 + sqrt(1 - alpha_cumprod[t]) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, cond=None):
        """
        完整 DDPM 采样循环: 从纯噪声 x_T 逐步去噪到 x_0
        """
        batch_size = shape[0]
        x_t = torch.randn(shape, device=device)

        # 条件广播
        cond_broadcast = cond
        if cond is not None and cond.shape[0] != batch_size:
            cond_broadcast = cond[:1].repeat(batch_size, 1, 1, 1) if cond.shape[0] == 1 else cond

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            betas_t = extract(self.betas, t, x_t.shape)
            sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            sqrt_recip_alphas = torch.sqrt(1.0 / extract(self.alphas, t, x_t.shape))

            if cond_broadcast is not None:
                predicted_noise = model(x_t, t, cond=cond_broadcast)
            else:
                predicted_noise = model(x_t, t)

            pred_mean = sqrt_recip_alphas * (
                x_t - betas_t / sqrt_one_minus * predicted_noise
            )

            if i == 0:
                x_t = pred_mean
            else:
                posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
                noise = torch.randn_like(x_t)
                x_t = pred_mean + torch.sqrt(posterior_variance_t) * noise

        return x_t

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, sampling_steps=50, eta=0.0, cond=None):
        """
        DDIM 采样（加速版）: 用更少的步数完成采样。
        eta=0 时为确定性采样（DDIM），eta>0 引入随机性。
        cond: 条件图（血管骨架等），用于条件扩散模型。
        """
        batch_size = shape[0]
        total_timesteps = self.timesteps

        # 选择采样步数对应的时刻
        step_indices = torch.linspace(0, total_timesteps - 1, sampling_steps, dtype=torch.long)
        step_indices = step_indices.flip(0)  # 从 T 到 0

        x_t = torch.randn(shape, device=device)

        # 条件广播（cond 应该和 batch 匹配）
        cond_broadcast = cond
        if cond is not None and cond.shape[0] != batch_size:
            cond_broadcast = cond[:1].repeat(batch_size, 1, 1, 1) if cond.shape[0] == 1 else cond

        for i, step in enumerate(step_indices):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            if cond_broadcast is not None:
                predicted_noise = model(x_t, t, cond=cond_broadcast)
            else:
                predicted_noise = model(x_t, t)

            # DDIM 更新公式
            alpha_cumprod_t = extract(self.alphas_cumprod, t, x_t.shape)
            alpha_cumprod_t_prev = extract(
                self.alphas_cumprod,
                torch.full((batch_size,), step_indices[i + 1] if i < sampling_steps - 1 else 0,
                          device=device, dtype=torch.long),
                x_t.shape
            )

            sigma = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            noise = torch.randn_like(x_t) if eta > 0 else 0

            x_t = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0 +
                torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * predicted_noise +
                sigma * noise
            )

        return x_t


# ============================================================
# UNet 模型（带 Self-Attention，改进版）
# 从 DenoisingDiffusionProbabilityModel-ddpm- 整合的注意力机制
# ============================================================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码：将时间步 t 编码为特征向量。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """时间步编码 MLP：将位置编码映射到 UNet 特征维度。"""

    def __init__(self, T, d_model, dim):
        super().__init__()
        assert d_model % 2 == 0
        # 预计算正余弦位置编码
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)


class AttnBlock(nn.Module):
    """自注意力块：用于捕获特征图上的长程依赖关系。"""

    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)
        self._init_weights()

    def _init_weights(self):
        for module in [self.proj_q, self.proj_k, self.proj_v]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).permute(0, 2, 3, 1).view(B, H * W, C)
        k = self.proj_k(h).view(B, C, H * W)
        v = self.proj_v(h).permute(0, 2, 3, 1).view(B, H * W, C)

        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    """
    残差块：GroupNorm + Swish + Conv + FiLM时间步调制 + zero_module + 可选注意力。

    FiLM 调制（Feature-wise Linear Modulation）来自 Palette / Guided-Diffusion：
    对 GroupNorm 后的特征逐通道做 h * (1+scale) + shift，让时间步信息能更精细地
    控制每个特征通道的增益，而非简单的通道加性偏置。这是保高频细节的关键改进。
    """

    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, use_scale_shift_norm=True):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm

        # Block 1: GN → Swish → Conv (in_ch → out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        # 时间步编码投影：2×out_ch（scale+shift）或 out_ch（旧版additive bias）
        temb_out_ch = 2 * out_ch if use_scale_shift_norm else out_ch
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, temb_out_ch),
        )

        # Block 2: GN → (FiLM) → SiLU → Dropout → zero_initialized Conv
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.silu2 = Swish()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # zero_module: 最后一层卷积初始化为 0，开始时残差分支 = 0
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x)

        # 时间步编码 → [B, C, 1, 1]
        temb_out = self.temb_proj(temb)[:, :, None, None]

        if self.use_scale_shift_norm:
            # FiLM 调制：GN(h) * (1+scale) + shift
            # 每个通道独立缩放+平移，时间步信息控制通道增益
            h = self.norm2(h)
            scale, shift = torch.chunk(temb_out, 2, dim=1)
            h = h * (1 + scale) + shift
            h = self.silu2(h)
            h = self.dropout2(h)
        else:
            # 旧版：简单加性偏置（兼容旧 checkpoint）
            h = h + temb_out
            h = self.norm2(h)
            h = self.silu2(h)
            h = self.dropout2(h)

        h = self.conv2(h)
        return self.attn(self.shortcut(x) + h)


class UNet(nn.Module):
    """
    UNet 架构用于扩散模型的噪声预测。
    包含下采样、上采样、跳跃连接、自注意力机制和时间步条件输入。
    注意: 为了在 128x128 输入上运行，下采样 4 次产生 8x8 特征图。
          如果调整 img_size，确保 img_size % 16 == 0。
    """

    def __init__(self, T=1000, channels=3, base_dim=128, dim_mults=(1, 2, 3, 4),
                 attn_layers=(2,), dropout=0.1, cond_channels=0,
                 use_scale_shift_norm=True):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        tdim = base_dim * 4
        self.time_embedding = TimeEmbedding(T, base_dim, tdim)

        # 条件输入处理
        self.cond_channels = cond_channels
        self.cond_proj = None
        if cond_channels > 0:
            self.cond_proj = nn.Conv2d(cond_channels, base_dim, 3, padding=1)
            nn.init.xavier_uniform_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

        # 初始卷积（RGB + 条件特征合并后的通道数）
        self.head = nn.Conv2d(channels, base_dim, 3, padding=1)

        # 下采样路径
        self.downblocks = nn.ModuleList()
        chs = [base_dim]
        now_ch = base_dim
        for i, mult in enumerate(dim_mults):
            out_ch = base_dim * mult
            for _ in range(2):  # 每层 2 个残差块
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn_layers),
                    use_scale_shift_norm=self.use_scale_shift_norm))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(dim_mults) - 1:
                # 下采样（stride=2 卷积）
                self.downblocks.append(nn.Conv2d(now_ch, now_ch, 3, stride=2, padding=1))
                chs.append(now_ch)

        # 中间层（带注意力）
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True,
                     use_scale_shift_norm=self.use_scale_shift_norm),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False,
                     use_scale_shift_norm=self.use_scale_shift_norm),
        ])

        # 上采样路径
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(dim_mults))):
            out_ch = base_dim * mult
            for _ in range(3):  # 每层 3 个残差块（比下采样多 1 个）
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn_layers),
                    use_scale_shift_norm=self.use_scale_shift_norm))
                now_ch = out_ch
            if i != 0:
                # 上采样（最近邻插值 + 卷积）
                self.upblocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(now_ch, now_ch, 3, padding=1),
                ))

        # 输出层
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, channels, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)
        # 重设 ResBlock 中 zero_module 的卷积为 0（全局 Xavier 会覆盖它）
        for module in self.modules():
            if isinstance(module, ResBlock):
                with torch.no_grad():
                    module.conv2.weight.zero_()
                    if module.conv2.bias is not None:
                        module.conv2.bias.zero_()

    def forward(self, x, t, cond=None):
        # 时间步编码
        temb = self.time_embedding(t)

        # 初始卷积
        h = self.head(x)

        # 条件特征处理（如果提供）
        if cond is not None and self.cond_proj is not None:
            h_cond = self.cond_proj(cond)
            h = h + h_cond  # 与 RGB 特征相加融合

        # 下采样 + 存储跳跃连接
        hs = [h]
        for layer in self.downblocks:
            if isinstance(layer, ResBlock):
                h = layer(h, temb)
            else:
                h = layer(h)  # 下采样
            hs.append(h)

        # 中间
        for layer in self.middleblocks:
            h = layer(h, temb)

        # 上采样
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)
            else:
                h = layer(h)  # 上采样

        return self.tail(h)


# ============================================================
# EMA（指数移动平均）
# 维护模型权重的平滑版本，采样时用 EMA 权重可显著提升生成质量。
# ============================================================
class EMA:
    """
    EMA（Exponential Moving Average）:
    在训练过程中维护模型参数的滑动平均。
    采样时切换到 EMA 参数，生成的图像通常更清晰、更稳定。
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 EMA 权重 = 当前模型权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """每一步训练后更新 EMA 权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """切换到 EMA 权重（采样前调用）"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """恢复原始权重（采样后调用）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """获取 EMA 权重用于保存"""
        return self.shadow

    def load_state_dict(self, state_dict):
        """加载 EMA 权重"""
        self.shadow = state_dict


# ============================================================
# 辅助函数
# ============================================================
def save_loss_plot(losses, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Diffusion Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_banner(args, device, dataset_len, cond_channels=0):
    print()
    print("=" * 60)
    print("Fundus-DDPM 训练开始")
    if cond_channels > 0:
        print(f"  {'='*10} 条件扩散模式 {'='*10}")
    print("=" * 60)
    print(f"  数据集路径:    {args.dataset_path}")
    print(f"  图片数量:      {dataset_len}")
    print(f"  输出目录:      {args.output_dir}")
    print(f"  设备:          {device}")
    print(f"  图像分辨率:    {args.img_size}x{args.img_size}")
    print(f"  批次大小:      {args.batch_size}")
    print(f"  训练轮数:      {args.epochs}")
    print(f"  学习率:        {args.lr}")
    print(f"  扩散步数:      {args.timesteps}")
    print(f"  UNet 基础通道: {args.base_dim}")
    print(f"  UNet 通道倍数: {args.dim_mults}")
    print(f"  注意力层:     {args.attn_layers}")
    print(f"  Dropout:       {args.dropout}")
    print(f"  FiLM 调制:     {'启用' if not args.no_scale_shift_norm else '禁用'}")
    print(f"  损失函数:      L1（替代 MSE，抗病灶溶解）")
    if hasattr(args, 'use_lpips') and args.use_lpips:
        print(f"  LPIPS 感知损失: 启用（权重={args.lpips_weight}）")
    if cond_channels > 0:
        print(f"  条件图路径:    {args.cond_path}")
        print(f"  条件通道数:    {cond_channels}")
    print(f"  模型保存间隔:  每 {args.model_save_interval} 轮")
    print(f"  图片保存间隔:  每 {args.image_save_interval} 轮")
    print(f"  预览网格:      {args.preview_grid_size}x{args.preview_grid_size}")
    print("=" * 60)
    print()


# ============================================================
# 主训练流程
# ============================================================
def main():
    args = parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    models_dir = os.path.join(args.output_dir, "models")
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 环境检查
    use_cuda = check_environment(args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # 数据预处理（增强版）
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    cond_channels = 1 if args.cond_path else 0
    cond_transform = None
    if args.cond_path:
        # 条件图的预处理（归一化到 [-1, 1]，不做 ColorJitter）
        cond_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    dataset = FundusDataset(
        args.dataset_path, transform=transform,
        cond_root=args.cond_path,
        img_size=args.img_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=min(args.num_workers, 4),
        pin_memory=use_cuda,
    )

    # 计算数据集 RGB 均值（用于颜色正则化）
    print("  计算数据集颜色统计...", end=" ", flush=True)
    dataset_means = []
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[0]
            dataset_means.append(batch_data.mean(dim=[0, 2, 3]))
    target_mean = torch.stack(dataset_means).mean(dim=0).to(device)
    print(f"目标 RGB 均值: [{target_mean[0]:.3f}, {target_mean[1]:.3f}, {target_mean[2]:.3f}]")

    print_banner(args, device, len(dataset), cond_channels=cond_channels)

    # 初始化扩散过程
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule="cosine",
    ).to(device)

    # 初始化 UNet（带注意力机制，可选条件输入，FiLM 调制）
    use_ssn = not args.no_scale_shift_norm
    model = UNet(
        T=args.timesteps,
        channels=args.channels,
        base_dim=args.base_dim,
        dim_mults=tuple(args.dim_mults),
        attn_layers=tuple(args.attn_layers),
        dropout=args.dropout,
        cond_channels=cond_channels,
        use_scale_shift_norm=use_ssn,
    ).to(device)

    n_params = count_parameters(model)
    print(f"  UNet 参数量: {n_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # EMA（指数移动平均）
    ema = EMA(model, decay=args.ema_decay)

    # LPIPS 感知损失（可选，需先 pip install lpips）
    lpips_fn = None
    if args.use_lpips:
        if _HAS_LPIPS:
            lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            print(f"  LPIPS 感知损失已加载（权重: {args.lpips_weight}）")
        else:
            print("  [WARN] lpips 未安装，跳过 LPIPS 损失（pip install lpips）")
            args.use_lpips = False

    # 断点续训（兼容新旧 ResBlock 架构）
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载 checkpoint: {args.resume}")
            cp = torch.load(args.resume, map_location=device, weights_only=False)
            old_state = cp["model_state_dict"]
            # 非严格加载：兼容新旧 ResBlock 结构差异（FiLM vs 加性偏置）
            missing, unexpected = model.load_state_dict(old_state, strict=False)
            if missing:
                print(f"  [INFO] {len(missing)} 个权重重新初始化（新旧 ResBlock 结构差异）:")
                for k in missing:
                    if 'temb_proj' not in k and 'conv2' not in k:
                        print(f"    - {k}")
            # 只加载匹配形状的优化器状态
            try:
                opt_state = cp["optimizer_state_dict"]
                # 过滤掉形状不匹配的参数
                model_params = dict(model.named_parameters())
                for k in list(opt_state['param_groups'][0]['params']):
                    pass  # params are stored as indices, not names
                optimizer.load_state_dict(opt_state)
                print(f"  优化器状态已加载")
            except Exception as e:
                print(f"  [WARN] 优化器状态不兼容（结构变化），已重新初始化: {e}")
            start_epoch = cp.get("epoch", -1) + 1
            print(f"  从 epoch {start_epoch} 继续训练")
            # 加载 EMA 权重（兼容旧 checkpoint）
            if "ema_state_dict" in cp:
                try:
                    ema.load_state_dict(cp["ema_state_dict"])
                    print(f"  EMA 权重已加载")
                except Exception as e:
                    print(f"  [WARN] EMA 权重不兼容，已重新初始化: {e}")
            else:
                print(f"  无 EMA 权重（旧版 checkpoint），从当前模型初始化 EMA")
        else:
            print("  checkpoint 不存在，从头开始")

    # 固定噪声用于生成预览
    fixed_noise = torch.randn(
        args.preview_grid_size ** 2, args.channels, args.img_size, args.img_size,
        device=device
    )

    # 条件扩散模式下：从数据集中取第一批条件图作为固定条件
    fixed_cond = None
    if args.cond_path:
        with torch.no_grad():
            for batch_data in dataloader:
                if batch_data[1] is not None:
                    cb = batch_data[1]
                    if isinstance(cb, (list, tuple)):
                        cb = torch.stack(cb)
                    # 确保预览图数量够用（不够就循环复制）
                    preview_n = args.preview_grid_size ** 2
                    if cb.shape[0] < preview_n:
                        repeats = (preview_n + cb.shape[0] - 1) // cb.shape[0]
                        cb = cb.repeat(repeats, 1, 1, 1)
                    fixed_cond = cb[:preview_n].to(device)
                break

    # 日志
    log_file = os.path.join(logs_dir, "training.log")
    log_mode = "a" if args.resume else "w"
    with open(log_file, log_mode, encoding="utf-8") as f:
        if log_mode == "w":
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Args: {json.dumps(vars(args))}\n\n")
        else:
            f.write(f"\n--- Resumed at epoch {start_epoch} ---\n")

    train_losses = []
    train_start_time = time.time()

    print("=" * 60)
    print("训练开始")
    print("提示: 按 Ctrl+C 可安全中断训练")
    print("=" * 60)
    print()

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            epoch_losses = []
            model.train()

            for batch_data in dataloader:
                real_data = batch_data[0].to(device, non_blocking=True)
                batch = real_data.shape[0]

                # 获取条件图（如果有）
                cond = None
                if args.cond_path and batch_data[1] is not None:
                    cond_batch = batch_data[1]
                    if isinstance(cond_batch, (list, tuple)):
                        cond_batch = torch.stack(cond_batch)
                    cond = cond_batch.to(device, non_blocking=True)

                # 随机采样时间步
                t = torch.randint(0, args.timesteps, (batch,), device=device, dtype=torch.long)

                # 前向加噪
                noisy_images, noise = diffusion.q_sample(real_data, t)

                # 预测噪声（条件模式下传入 cond）
                if cond is not None:
                    predicted_noise = model(noisy_images, t, cond=cond)
                else:
                    predicted_noise = model(noisy_images, t)

                # L1 损失（噪声预测，替代 MSE — L1 对病灶等异常值更鲁棒，缓解"溶解"效应）
                loss = F.l1_loss(predicted_noise, noise)

                # 颜色正则化 + LPIPS：只在低噪声 timestep 上约束
                # t 越小，噪声越少，反推的 x_0 越可靠
                low_noise_mask = t < 200
                if low_noise_mask.any():
                    t_low = t[low_noise_mask]
                    x_t_low = noisy_images[low_noise_mask]
                    pred_noise_low = predicted_noise[low_noise_mask]

                    sqrt_alpha = extract(diffusion.sqrt_alphas_cumprod, t_low, x_t_low.shape)
                    sqrt_one_minus = extract(diffusion.sqrt_one_minus_alphas_cumprod, t_low, x_t_low.shape)

                    # 从噪声预测反推原始图估计
                    x_0_pred = (x_t_low - sqrt_one_minus * pred_noise_low) / sqrt_alpha
                    x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

                    # 计算这批预测图的 RGB 均值与目标均值的差距
                    pred_mean = x_0_pred.mean(dim=[0, 2, 3])
                    color_loss = F.mse_loss(pred_mean, target_mean)

                    # 颜色正则权重（小权重，不干扰主任务）
                    loss = loss + args.color_weight * color_loss

                    # LPIPS 感知损失：在像素空间衡量"看起来像不像"
                    if lpips_fn is not None:
                        real_low = real_data[low_noise_mask]
                        lpips_val = lpips_fn(x_0_pred, real_low, normalize=True).mean()
                        loss = loss + args.lpips_weight * lpips_val

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                # 更新 EMA
                ema.update(model)

                epoch_losses.append(loss.item())

            # 记录
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            epoch_time = time.time() - epoch_start

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [{epoch+1:>{len(str(args.epochs))}}/{args.epochs}]  "
                      f"loss={avg_loss:.6f}  [{epoch_time:.1f}s]", flush=True)

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch+1}: loss={avg_loss:.6f}, time={epoch_time:.1f}s\n")

            # 保存模型
            if (epoch + 1) % args.model_save_interval == 0 or epoch == args.epochs - 1:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                    "has_ema": True,
                }, os.path.join(models_dir, f"checkpoint_epoch_{epoch+1:06d}.pth"))
                print(f"  [SAVE] 模型已保存: epoch {epoch+1}")

            # 生成预览图（使用 EMA 权重）
            if (epoch + 1) % args.image_save_interval == 0:
                model.eval()
                # 切换到 EMA 权重
                ema.apply_shadow(model)
                print(f"  [GEN] 正在生成预览图 (DDIM, 100步, EMA)...", flush=True)
                with torch.no_grad():
                    sample_shape = (args.preview_grid_size ** 2, args.channels, args.img_size, args.img_size)
                    all_samples = []
                    sample_batch = min(4, args.preview_grid_size ** 2)
                    n_batches = (args.preview_grid_size ** 2 + sample_batch - 1) // sample_batch
                    for i in range(n_batches):
                        bs = min(sample_batch, args.preview_grid_size ** 2 - i * sample_batch)
                        shape = (bs, args.channels, args.img_size, args.img_size)
                        # 条件采样
                        cond_i = None
                        if fixed_cond is not None:
                            start_idx = i * sample_batch
                            end_idx = min(start_idx + bs, fixed_cond.shape[0])
                            cond_i = fixed_cond[start_idx:end_idx]
                        samples = diffusion.ddim_sample(model, shape, device, sampling_steps=100,
                                                        cond=cond_i)
                        all_samples.append(samples.cpu())
                    samples = torch.cat(all_samples, dim=0)

                save_image(samples, os.path.join(images_dir, f"epoch_{epoch+1:06d}.png"),
                           nrow=args.preview_grid_size, normalize=True)
                print(f"  [IMG] 预览图已保存: epoch {epoch+1}", flush=True)
                # 恢复原始权重
                ema.restore(model)
                model.train()

            # Loss 曲线
            if (epoch + 1) % max(args.image_save_interval // 2, 1) == 0:
                save_loss_plot(train_losses, os.path.join(logs_dir, "loss_curve.png"))

    except KeyboardInterrupt:
        print("\n\n[WARN] 训练被中断", flush=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args,
            "has_ema": True,
        }, os.path.join(models_dir, "checkpoint_INTERRUPTED.pth"))
        print("  [SAVE] 已保存中断时的模型（含 EMA 和优化器状态）", flush=True)

    total_time = time.time() - train_start_time
    print()
    print("=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  总耗时: {total_time/60:.1f} 分钟")

    # 保存最终模型
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args,
        "has_ema": True,
    }, os.path.join(models_dir, "final_model.pth"))
    print(f"  最终模型 -> {os.path.join(models_dir, 'final_model.pth')}")

    save_loss_plot(train_losses, os.path.join(logs_dir, "loss_curve_final.png"))
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")
    print()


if __name__ == "__main__":
    main()
