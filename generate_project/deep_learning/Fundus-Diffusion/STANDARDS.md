# Fundus-Diffusion 项目规范

## 概述
本项目实现扩散模型（Diffusion Models）用于眼底彩照生成。

## 实现模型
- `ddpm/` — Denoising Diffusion Probabilistic Models
- `ddim/` — Denoising Diffusion Implicit Models（DDPM 加速版）
- `cond_diffusion/` — Conditional Diffusion（条件控制生成）

## 统一 CLI
遵循 `docs/03-Development-Standards.md` 定义的 CLI 规范。

## Diffusion 特有参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--timesteps` | int | 1000 | 扩散步数（训练用） |
| `--sampling_steps` | int | 50 | 采样步数（DDIM 加速用） |
| `--beta_start` | float | 1e-4 | 噪声调度起始值 |
| `--beta_end` | float | 0.02 | 噪声调度结束值 |
| `--lr` | float | 1e-4 | 学习率 |

## 训练命令示例
```bash
# DDPM
python ddpm/train.py --epochs 500 --batch_size 16 --img_size 128 --timesteps 1000 \
    --dataset_path "./fundus/_all_images_ORIGINAL" --output_dir "./results/ddpm"

# DDIM (使用 DDPM 权重 + DDIM 采样)
python ddim/train.py --epochs 500 --batch_size 16 --img_size 128 \
    --timesteps 1000 --sampling_steps 50 \
    --dataset_path "./fundus/_all_images_ORIGINAL" --output_dir "./results/ddim"
```

## 评估重点
- 生成图像的真实感和细节丰富度
- 采样速度（DDPM vs DDIM 对比）
- 不同去噪步数对质量的影响
