# Fundus-GAN 项目规范

## 概述
本项目实现 GAN（Generative Adversarial Networks）及其变体用于眼底彩照生成。

## 实现模型
- `dcgan/` — Deep Convolutional GAN（基础 GAN）
- `wgan_gp/` — Wasserstein GAN with Gradient Penalty（更稳定）
- `stylegan2/` — StyleGAN2-ADA（NVIDIA SOTA，适合小数据集）

## 统一 CLI
遵循 `docs/03-Development-Standards.md` 定义的 CLI 规范。

## GAN 特有参数（通用）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--latent_dim` | int | 100 | 噪声向量维度 |
| `--g_lr` | float | None | 生成器学习率（默认同 --lr） |
| `--d_lr` | float | None | 判别器学习率（默认同 --lr） |
| `--g_steps` | int | 1 | 每轮 G 训练次数 |
| `--d_steps` | int | 1 | 每轮 D 训练次数 |

## DCGAN 特有参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--instance_noise` | float | 0.1 | D 输入高斯噪声强度 |
| `--r1_gamma` | float | 0 | R1 梯度惩罚系数 |
| `--label_noise` | float | 0 | 标签噪声概率 |

## WGAN-GP 特有参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--n_critic` | int | 5 | D 训练次数 / G 训练 1 次 |
| `--lambda_gp` | float | 10.0 | Gradient Penalty 系数 |

## 训练命令示例
```bash
# DCGAN
python dcgan/train.py --epochs 800 --batch_size 16 --img_size 128 \
    --dataset_path "./fundus/_all_images_ORIGINAL" --output_dir "./results/dcgan" \
    --instance_noise 0.1 --r1_gamma 10

# WGAN-GP
python wgan_gp/train.py --epochs 800 --batch_size 16 --img_size 128 --lr 0.0001 \
    --dataset_path "./fundus/_all_images_ORIGINAL" --output_dir "./results/wgan_gp"

# StyleGAN2-ADA
python stylegan2/train.py --epochs 800 --batch_size 16 --img_size 128 \
    --dataset_path "./fundus/_all_images_ORIGINAL" --output_dir "./results/stylegan2"
```
