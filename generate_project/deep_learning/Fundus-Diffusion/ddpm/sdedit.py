"""
SDEdit 自精修脚本（DL-EX-023 第 2 步）
====================================
对模型自己生成图做低强度 img2img：加噪到 t0 再 DDIM 去噪回来，
重掷中高频细节（白点/噪点）、保留全局布局。依据 Meng et al., ICLR 2022 (arXiv:2108.01073)。

流程（全内存，无 PNG 往返损失）：
  1. 同 seed/条件在内存复现 N 张底图（与 generate.py 判定图完全同源）
  2. 每档 t0：x_t0 = q_sample(x0, t0)（固定噪声），从 t0 起步 DDIM 去噪到 0
  3. 输出 orig/ 与 t0xxx/ 单张 + 四列对照网格 + 白像素率前后对比

用法:
    python sdedit.py --checkpoint results_D3_combined128_finetune/models/final_model.pth \
        --cond_path ./conditions_combined_128 --num_images 8 --t0 200 300 400 \
        --output_dir ./sdedit_D3_ema
"""

import argparse
import os

import torch
from torchvision.utils import save_image

from train import UNet, GaussianDiffusion, check_environment, extract
from generate import load_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="SDEdit 自精修（低强度 img2img）")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cond_path", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--base_dim", type=int, default=128)
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--attn_layers", type=int, nargs="+", default=[2])
    parser.add_argument("--t0", type=int, nargs="+", default=[200, 300, 400],
                        help="加噪起点档位（0-999，越大改动越大）")
    parser.add_argument("--denoise_steps", type=int, default=100, help="t0→0 的 DDIM 步数")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--gen_seed", type=int, default=42, help="底图生成 seed（复现判定图）")
    parser.add_argument("--noise_seed", type=int, default=1234, help="q_sample 加噪 seed（跨档位复用同一噪声）")
    parser.add_argument("--output_dir", type=str, default="./sdedit_out")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def ddim_sample_from(model, diffusion, x_t0, t0, steps, eta, cond):
    """从 x_t0（位于 t0 时刻）出发的 DDIM 去噪，步数均布在 [0, t0]。逻辑同 train.ddim_sample。"""
    batch_size = x_t0.shape[0]
    step_indices = torch.linspace(t0, 0, steps, dtype=torch.long, device=x_t0.device)
    # linspace(t0,0,N) 已从高到低；去重保证严格递减（t0<steps-1 时可能重复）
    dedup = []
    for s in step_indices.tolist():
        if not dedup or s < dedup[-1]:
            dedup.append(s)
    step_indices = torch.tensor(dedup, device=x_t0.device, dtype=torch.long)

    x_t = x_t0
    for i, step in enumerate(step_indices):
        t = torch.full((batch_size,), step, device=x_t0.device, dtype=torch.long)
        predicted_noise = model(x_t, t, cond=cond) if cond is not None else model(x_t, t)

        alpha_t = extract(diffusion.alphas_cumprod, t, x_t.shape)
        prev_step = step_indices[i + 1].item() if i < len(step_indices) - 1 else 0
        alpha_prev = extract(
            diffusion.alphas_cumprod,
            torch.full((batch_size,), prev_step, device=x_t0.device, dtype=torch.long),
            x_t.shape,
        )

        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        noise = torch.randn_like(x_t) if eta > 0 else 0
        x_t = (
            torch.sqrt(alpha_prev) * pred_x0
            + torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise
            + sigma * noise
        )
    return x_t


def white_pixel_rate(images):
    """白像素率：[0,1] 空间 >0.95 的像素占比（白点伪影代理指标）。images: [B,C,H,W] in [-1,1]"""
    x = (images.clamp(-1, 1) + 1) / 2
    return (x > 0.95).float().mean(dim=(1, 2, 3))  # [B]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if check_environment(args.no_cuda) else "cpu")
    cond_channels = 1 if args.cond_path else 0

    # ---- 模型加载（与 generate.py 同逻辑：读架构参数 + EMA 默认）----
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    use_ssn = getattr(ckpt_args, "use_scale_shift_norm", True) if ckpt_args is not None else True
    model = UNet(
        T=1000, channels=args.channels, base_dim=args.base_dim,
        dim_mults=tuple(args.dim_mults), attn_layers=tuple(args.attn_layers),
        dropout=0.0, cond_channels=cond_channels, use_scale_shift_norm=use_ssn,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if "ema_state_dict" in checkpoint and checkpoint.get("has_ema", False):
        model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
        print("[OK] EMA 权重")
    else:
        print("[OK] 原始权重")
    model.to(device).eval()

    # ---- 条件图（与 generate.py 同序：sorted 后循环取前 N）----
    conds = None
    if args.cond_path:
        conds = load_condition_images(args.cond_path, args.img_size, args.num_images).to(device)

    diffusion = GaussianDiffusion(timesteps=1000).to(device)

    # ---- 1. 内存复现底图（同 seed 同条件 = 判定图同源）----
    torch.manual_seed(args.gen_seed)
    shape = (args.num_images, args.channels, args.img_size, args.img_size)
    print(f"[GEN] DDIM 100 复现底图 {args.num_images} 张（seed {args.gen_seed}）...")
    with torch.no_grad():
        x0 = diffusion.ddim_sample(model, shape, device, sampling_steps=100,
                                   eta=0.0, cond=conds)
    orig_dir = os.path.join(args.output_dir, "orig")
    os.makedirs(orig_dir, exist_ok=True)
    for i in range(args.num_images):
        save_image(x0[i], os.path.join(orig_dir, f"sample_{i:04d}.png"), normalize=True)

    # ---- 2. 各档位 SDEdit（跨档位复用同一份加噪噪声）----
    torch.manual_seed(args.noise_seed)
    noise = torch.randn_like(x0)

    refined = {"orig": x0}
    for t0 in args.t0:
        print(f"[SDEDIT] t0={t0}（{args.denoise_steps} 步去噪）...")
        with torch.no_grad():
            t_batch = torch.full((args.num_images,), t0, device=device, dtype=torch.long)
            x_t0, _ = diffusion.q_sample(x0, t_batch, noise=noise)  # q_sample 返回 (x_t, noise)
            refined[t0] = ddim_sample_from(model, diffusion, x_t0, t0,
                                           args.denoise_steps, args.eta, conds)
        t_dir = os.path.join(args.output_dir, f"t0{t0:04d}")
        os.makedirs(t_dir, exist_ok=True)
        for i in range(args.num_images):
            save_image(refined[t0][i], os.path.join(t_dir, f"sample_{i:04d}.png"), normalize=True)

    # ---- 3. 对照网格：每行 [orig | t0_1 | t0_2 | ...] ----
    keys = ["orig"] + list(args.t0)
    rows = []
    for i in range(args.num_images):
        row = [refined[k][i] for k in keys]
        rows.append(torch.stack(row, dim=0))
    grid = torch.cat(rows, dim=0)
    save_image(grid, os.path.join(args.output_dir, "compare_grid.png"),
               nrow=len(keys), normalize=True)

    # ---- 4. 白像素率前后对比 ----
    print("\n白像素率（>0.95 占比，%）：")
    print(f"{'图':>4} | " + " | ".join(f"{k if isinstance(k, str) else f't0={k}'}" for k in keys))
    rates = {k: white_pixel_rate(v) for k, v in refined.items()}
    for i in range(args.num_images):
        print(f"{i:>4} | " + " | ".join(f"{rates[k][i].item() * 100:.3f}" for k in keys))
    mean_row = " | ".join(f"{rates[k].mean().item() * 100:.3f}" for k in keys)
    print(f"均值 | {mean_row}")

    print(f"\n[OK] 完成：{args.output_dir}/（orig、t0 各档、compare_grid.png）")


if __name__ == "__main__":
    main()
