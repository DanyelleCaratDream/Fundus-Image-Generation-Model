"""
Fundus-Diffusion DDPM/DDIM 生成脚本
====================================
用训练好的扩散模型生成眼底彩照。
支持 DDPM 采样（高质量）和 DDIM 采样（快速）两种方式。
支持条件扩散模式（输入血管骨架图 + 噪声 → 输出眼底彩照）。

用法:
    # DDIM 采样（50步，快20倍，推荐）
    python generate.py --checkpoint "./results/models/final_model.pth" \
        --num_images 64 --output_dir "./generated" --sampler ddim --sampling_steps 50

    # 条件扩散采样（给定血管骨架生成眼底图）
    python generate.py --checkpoint "./results/models/cond_model.pth" \
        --cond_path "./conditions" --num_images 16 --output_dir "./generated_cond" \
        --sampler ddim --sampling_steps 100

    # 带颜色校正的生成（修复颜色偏差）
    python generate.py --checkpoint "./results/models/final_model.pth" \
        --num_images 64 --output_dir "./generated_colorfix" \
        --sampler ddim --sampling_steps 100 \
        --color_correct --dataset_path "../../../../fundus/_all_images_ORIGINAL"
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from train import UNet, GaussianDiffusion, check_environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fundus-Diffusion: Generate fundus images from trained diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--num_images", type=int, default=64, help="生成图片数量")
    parser.add_argument("--img_size", type=int, default=128, help="图像分辨率")
    parser.add_argument("--channels", type=int, default=3, help="图像通道数")
    parser.add_argument("--base_dim", type=int, default=128, help="UNet 基础通道数")
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="UNet 每层通道数倍数")
    parser.add_argument("--attn_layers", type=int, nargs="+", default=[2],
                        help="注意力层")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout（推理时 = 0）")
    parser.add_argument("--output_dir", type=str, default="./generated", help="输出文件夹")
    parser.add_argument("--grid_size", type=int, default=8, help="网格大小 (0=单张)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="强制CPU")

    # 采样参数
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"],
                        help="采样方式: ddpm=高质量慢速, ddim=快速")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="DDIM 采样步数（越小越快，建议 20-100）")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM 随机性 (0=确定性, >0=有随机性)")

    # 条件扩散参数
    parser.add_argument("--cond_path", type=str, default=None,
                        help="条件图路径（血管骨架等），开启条件扩散推理")

    # 颜色校正参数
    parser.add_argument("--color_correct", action="store_true",
                        help="用真实数据集 RGB 统计量做颜色校正（修复颜色偏差）")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="真实数据集路径，用于计算 RGB 统计量（color_correct 需要）")

    return parser.parse_args()


def load_condition_images(cond_path, img_size=128, num_images=64):
    """从文件夹加载条件图（血管骨架），返回归一化后的 tensor [B, 1, H, W]"""
    if not os.path.isdir(cond_path):
        print(f"[ERROR] 条件图路径不存在: {cond_path}")
        sys.exit(1)

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    import glob
    cond_files = []
    for ext in exts:
        cond_files.extend(glob.glob(os.path.join(cond_path, ext)))
    cond_files = sorted(cond_files)

    if not cond_files:
        print(f"[ERROR] 在 {cond_path} 中没有找到条件图")
        sys.exit(1)

    # 取前 num_images 张（或循环使用）
    cond_tensors = []
    for i in range(num_images):
        f = cond_files[i % len(cond_files)]
        try:
            img = Image.open(f).convert("L").resize((img_size, img_size))
            t = (torch.from_numpy(np.array(img)).float() / 255.0 * 2.0 - 1.0).unsqueeze(0)
            cond_tensors.append(t)
        except Exception as e:
            print(f"  [WARN] 加载条件图失败: {f} ({e})")
            cond_tensors.append(torch.zeros(1, img_size, img_size))

    conds = torch.stack(cond_tensors)  # [B, 1, H, W]
    print(f"[OK] 已加载 {len(cond_files)} 张条件图，循环使用生成 {num_images} 张")
    return conds


def compute_dataset_rgb_stats(dataset_path, img_size=128):
    """遍历数据集，计算 RGB 通道的均值和标准差（在 [-1, 1] 空间）。"""
    import glob
    from PIL import Image

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(dataset_path, ext)))
    files = sorted(files)

    if not files:
        print(f"[ERROR] 数据集路径为空: {dataset_path}")
        return None, None

    print(f"[STATS] 正在计算 {len(files)} 张图的 RGB 统计量...")
    pixels = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((img_size, img_size))
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # -> [-1, 1]
        pixels.append(arr.reshape(-1, 3))
    pixels = np.concatenate(pixels, axis=0)  # [N, 3]

    mean = pixels.mean(axis=0)  # [3]
    std = pixels.std(axis=0)    # [3]
    print(f"  RGB 均值: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  RGB 标准差: [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    return mean, std


def apply_color_correct(images, target_mean, target_std):
    """
    将每张图的 RGB 通道缩放到目标均值和标准差。
    images: [B, 3, H, W]  in [-1, 1]
    target_mean: [3], target_std: [3]
    """
    B, C, H, W = images.shape
    target_mean = torch.tensor(target_mean, device=images.device).view(1, 3, 1, 1)
    target_std = torch.tensor(target_std, device=images.device).view(1, 3, 1, 1)

    # 每张图自己的 RGB 统计量
    img_mean = images.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)  # [B, 3, 1, 1]
    img_std = images.view(B, C, -1).std(dim=2).view(B, C, 1, 1)    # [B, 3, 1, 1]
    img_std = torch.clamp(img_std, min=1e-6)

    corrected = (images - img_mean) / img_std * target_std + target_mean
    return corrected


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] 模型文件不存在: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    cond_channels = 1 if args.cond_path else 0

    use_cuda = check_environment(args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # 从 checkpoint 检测是否使用 FiLM 调制
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    use_ssn = True
    if ckpt_args is not None:
        use_ssn = getattr(ckpt_args, "use_scale_shift_norm", True)
        if hasattr(ckpt_args, "no_scale_shift_norm") and ckpt_args.no_scale_shift_norm:
            use_ssn = False

    # 兼容旧版 UNet checkpoint（block2 结构，无 FiLM）：自动重映射 key 并禁用 FiLM
    # 旧版 ResBlock: block1 → block2(GN→Swish→Dropout→Conv)；新版: block1 → norm2/silu2/dropout2/conv2
    def _remap_legacy(sd):
        out = {}
        for k, v in sd.items():
            if ".block2.0." in k:
                k = k.replace(".block2.0.", ".norm2.")
            elif ".block2.3." in k:
                k = k.replace(".block2.3.", ".conv2.")
            out[k] = v
        return out

    legacy_state = checkpoint.get("model_state_dict", checkpoint)
    is_legacy = any("block2.0." in k for k in legacy_state)
    if is_legacy:
        use_ssn = False
        print("[INFO] 检测到旧版 UNet checkpoint（block2 结构）→ 自动禁用 FiLM 并重映射 key")
        if "model_state_dict" in checkpoint:
            checkpoint["model_state_dict"] = _remap_legacy(checkpoint["model_state_dict"])
            if "ema_state_dict" in checkpoint:
                checkpoint["ema_state_dict"] = _remap_legacy(checkpoint["ema_state_dict"])
        else:
            checkpoint = _remap_legacy(checkpoint)

    print(f"[INFO] 模型架构: FiLM 调制 = {'启用' if use_ssn else '禁用'}")

    # 加载模型（支持条件扩散 + FiLM）
    model = UNet(
        T=1000,
        channels=args.channels,
        base_dim=args.base_dim,
        dim_mults=tuple(args.dim_mults),
        attn_layers=tuple(args.attn_layers),
        dropout=args.dropout,
        cond_channels=cond_channels,
        use_scale_shift_norm=use_ssn,
    )

    # 兼容不同保存格式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 兼容条件/无条件权重切换（如果 checkpoint 无条件但模型有条件，忽略 cond_proj 权重）
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    if missing_keys and cond_channels > 0:
        print(f"[INFO] 缺少 {len(missing_keys)} 个权重（条件模块未训练?）")
    if missing_keys and cond_channels == 0:
        # 可能是旧版 checkpoint 不兼容 FiLM，提示用户
        film_keys = [k for k in missing_keys if 'norm2' in k or 'silu2' in k or 'conv2' in k]
        if film_keys:
            print(f"[WARN] 检测到 ResBlock 结构不匹配（{len(film_keys)} 个层），")
            print(f"       这通常是因为 checkpoint 用旧版训练而模型启用了 FiLM。")
            print(f"       请尝试添加 --no_scale_shift_norm 参数重试。")

    model.to(device)
    model.eval()

    # 优先使用 EMA 权重
    if "ema_state_dict" in checkpoint and checkpoint.get("has_ema", False):
        ema_state = checkpoint["ema_state_dict"]
        model.load_state_dict(ema_state, strict=False)
        print(f"[OK] 已加载模型（EMA 权重）: {args.checkpoint}")
    else:
        print(f"[OK] 已加载模型（原始权重）: {args.checkpoint}")

    # 加载条件图
    conds = None
    if args.cond_path:
        conds = load_condition_images(args.cond_path, args.img_size, args.num_images).to(device)
        print(f"[INFO] 条件图张量形状: {conds.shape}")

    # 初始化扩散过程
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    print(f"[INFO] 采样方式: {args.sampler.upper()}")

    # 生成
    print(f"[GEN] 正在生成 {args.num_images} 张图片...")
    batch_size = min(args.num_images, 8)
    all_images = []

    with torch.no_grad():
        remaining = args.num_images
        batch_idx = 0
        while remaining > 0:
            bs = min(batch_size, remaining)
            shape = (bs, args.channels, args.img_size, args.img_size)

            # 取对应的条件图
            cond_batch = None
            if conds is not None:
                cond_batch = conds[batch_idx * batch_size: batch_idx * batch_size + bs]

            if args.sampler == "ddpm":
                print(f"  DDPM 采样 ({bs} 张, 1000 步)...", end=" ", flush=True)
                samples = diffusion.p_sample_loop(model, shape, device, cond=cond_batch)
            else:
                print(f"  DDIM 采样 ({bs} 张, {args.sampling_steps} 步)...", end=" ", flush=True)
                samples = diffusion.ddim_sample(
                    model, shape, device,
                    sampling_steps=args.sampling_steps,
                    eta=args.eta,
                    cond=cond_batch,
                )

            print("完成")
            all_images.append(samples.cpu())
            remaining -= bs
            batch_idx += 1

    all_images = torch.cat(all_images, dim=0)
    print(f"[OK] 已生成 {len(all_images)} 张图片")

    # 颜色校正（对齐到真实数据集 RGB 统计量）
    if args.color_correct:
        if not args.dataset_path:
            print("[ERROR] --color_correct 需要 --dataset_path 指定真实数据集路径")
            sys.exit(1)
        real_mean, real_std = compute_dataset_rgb_stats(args.dataset_path, args.img_size)
        if real_mean is not None:
            all_images = apply_color_correct(all_images, real_mean, real_std)
            print(f"[OK] 颜色校正完成")

    # 保存网格图
    if args.grid_size > 0:
        grid_path = os.path.join(args.output_dir, f"grid_{args.num_images}.png")
        save_image(all_images, grid_path, nrow=args.grid_size, normalize=True)
        print(f"[IMG] 网格图已保存: {grid_path}")

    # 保存单张
    singles_dir = os.path.join(args.output_dir, "singles")
    os.makedirs(singles_dir, exist_ok=True)
    for idx in range(min(args.num_images, len(all_images))):
        save_image(all_images[idx], os.path.join(singles_dir, f"sample_{idx:04d}.png"), normalize=True)
    print(f"[IMG] 单张图已保存到: {singles_dir}")
    print("[OK] 生成完成！")


if __name__ == "__main__":
    main()
