# Fundus-VAE 项目规范

> ⚠️ **现状更正（2026-10-15）**：上方列出的 `vanilla_vae/`、`beta_vae/` 两个子目录**从未建立**——
> `train.py` / `generate.py` 就在本目录下，不同实验靠 `--output_dir` 区分；beta-VAE 当时未实际训练。
> 下方 VAE 特有参数表**仍然准确**；实际命令见 `docs/08-Work-Guide.md`。

## 概述
本项目实现 VAE（Variational Autoencoder）及其变体用于眼底彩照生成。

## 实现模型
- `vanilla_vae/` — 标准 VAE
- `beta_vae/` — beta-VAE（增加 KL 权重以促进隐空间解耦）

## 统一 CLI
遵循 `docs/03-Development-Standards.md` 定义的 CLI 规范。

## VAE 特有参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--latent_dim` | int | 128 | 隐空间维度 |
| `--beta` | float | 1.0 | KL 权重（>1 为 beta-VAE） |
| `--dim` | int | 32 | 基础通道数 |

## 训练命令示例
```bash
# vanilla VAE
python vanilla_vae/train.py --epochs 500 --batch_size 32 --img_size 128 \
    --dataset_path "./fundus/_all_images_ORIGINAL" --output_dir "./results/vanilla_vae"

# beta-VAE (beta=4)
python beta_vae/train.py --epochs 500 --batch_size 32 --img_size 128 --beta 4 \
    --dataset_path "./fundus/_all_images_ORIGINAL" --output_dir "./results/beta_vae"
```

## 评估
VAE 的评估重点：
- 重建质量（reconstruction fidelity）
- 生成样本的多样性
- 隐空间插值的平滑度
