# FundusGen — 眼底彩照生成模型研究

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-13.0+-76b900.svg)]()

> **系统性探索 VAE → GAN → Diffusion 三大生成家族在眼底彩照生成任务上的表现。**
> 330 张小样本场景，8GB VRAM，从 VAE 模糊天花板到 FiLM DDPM + L1 + LPIPS 最佳实践的完整实验记录。

---

## 📋 项目概述

本项目以 **眼底彩照生成** 为目标，在仅 330 张严重症状眼底彩照的小样本设定下，依次探索了 VAE、GAN、Diffusion 三大生成模型家族。每个模型均经历了训练、评估、失败分析的过程，最终在 **FiLM 调制 DDPM + L1 损失 + LPIPS 感知损失** 的组合上取得最佳效果（85-90/100 分）。

研究路线（老师指定，从后往前）：**VAE → GAN → Diffusion → Flow Matching**（Flow Matching 纳入展望）。

### 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) |
| CPU | Intel/AMD 笔记本处理器 |
| 内存 | 16 GB |

---

## 📁 项目结构

```
AI_Model_Project_for_Fundus_Color_Images/
├── docs/                           # 规范文档（项目概述、技术栈、开发标准等）
├── Fundus-VAE/                     # VAE 系列
│   ├── train.py                    #   训练入口
│   ├── generate.py                 #   推理入口
│   ├── vanilla_vae/                #   基本 VAE
│   ├── beta_vae/                   #   beta-VAE（含 KL 加权）
│   └── results/                    #   训练结果（生成图 + 重建图）
├── Fundus-GAN/                     # GAN 系列
│   ├── dcgan/                      #   DCGAN
│   │   ├── train.py
│   │   └── generate.py
│   ├── wgan_gp/                    #   WGAN-GP（梯度惩罚）
│   │   ├── train.py
│   │   └── generate.py
│   └── stylegan2/                  #   StyleGAN2-ADA（小样本自适应）
│       ├── train.py
│       └── generate.py
├── Fundus-Diffusion/               # Diffusion 系列 ⭐ 主力
│   ├── ddpm/                       #   DDPM + 条件扩散
│   │   ├── train.py                #   训练脚本（含 FiLM / L1+LPIPS）
│   │   ├── generate.py             #   推理采样
│   │   ├── prep_conditions.py      #   条件图（血管骨架）预处理
│   │   ├── conditions/             #   血管骨架条件图（330 张）
│   │   └── results_film_l1lpips/   #   最佳实验结果
│   └── STANDARDS.md
├── research-report/                # 科研报告
│   ├── REPORT.docx                 #   完整实验报告（含结论章）
│   ├── generate_docx.py            #   DOCX 生成脚本
│   ├── ref_report.txt              #   参考报告文本
│   └── *.png                       #   报告嵌入结果图
├── fundus/                         # 数据集（git 排除，需单独获取）
├── .gitignore
├── CONTINUE_GUIDE_NEW.md           # 实验继续指引
└── README.md                       # 本文件
```

---

## 🧪 模型实验全记录

### 1️⃣ VAE（基线）— 10/100

| 模型 | 评分 | 关键问题 |
|------|:----:|----------|
| Vanilla VAE | 10/100 | MSE 损失导致严重模糊，"模糊天花板"效应 |
| Beta-VAE | 10/100 | 增加 KL 权重后重建更模糊 |

**结论**：VAE 在像素级损失下的模糊天花板无法突破，需转向更强大的生成模型。

### 2️⃣ GAN 系列 — 20/100

| 模型 | 评分 | 关键问题 |
|------|:----:|----------|
| DCGAN | 20/100 | 判别器在小样本下迅速过拟合，生成图像伪影严重 |
| WGAN-GP | 20/100 | 梯度惩罚未能解决过拟合，训练震荡剧烈 |
| StyleGAN2-ADA | — | CUDA 环境不兼容，未完成训练 |

**核心失败分析**：330 张样本对 GAN 的判别器来说严重不足。ADA 机制理论上适合小样本，但环境配置受阻。

### 3️⃣ Diffusion 系列 ⭐

| 模型 | 评分 | 说明 |
|------|:----:|------|
| 无条件 DDPM | 75/100 | 结构正确但缺少细节纹理 |
| 条件 DDPM (MSE) | 70/100 | 220 轮巅峰后病灶逐渐溶解 |
| Palette (MSE) | 0/100 | 125 轮巅峰 → 275 模糊 → 推理完全失败 |
| **FiLM DDPM (MSE, 500轮)** | **85/100** | **FiLM 调制延缓溶解到 450 轮** |
| **FiLM DDPM + L1 + LPIPS (780轮)** | **85~90/100** | **🏆 最佳方案** |

#### 关键技术发现

**MSE 溶解（MSE Dissolution）**：MSE 损失在暗背景（95% 像素）上优化时，亮病灶区域的梯度贡献被大量暗像素平均化，导致病灶特征随时间推移逐渐模糊。这是扩散模型在小样本眼底图上的独特现象。

**FiLM 调制**：特征线性调制（Feature-wise Linear Modulation）—— `GN(h) * (1+scale) + shift` —— 使 UNet 在每个残差块中根据时间步动态调整特征，显著延缓了溶解速度。

**L1 + LPIPS 组合**（最佳方案）：
- L1 噪声预测损失（等价 MSE ~0.0025），对离群点更鲁棒
- LPIPS 感知损失（权重 0.1，仅 t<200 时间步），基于预训练 AlexNet 的特征相似度
- EMA decay 0.9999（保留更多历史信息）
- 交替训练策略：ORIGINAL(330张) ↔ much(1320张增强版) 每 60-180 轮切换

**最佳效果**：~700 轮，病灶保持良好，血管纹理锐利。

---

## 🚀 快速开始

### 环境要求

```bash
pip install torch torchvision numpy Pillow matplotlib
pip install lpips  # 感知损失（最佳方案需要）
```

### 训练（DDPM）

```bash
cd Fundus-Diffusion/ddpm

# 基础 DDPM 训练
python train.py --epochs 500 --batch_size 16 --img_size 128 \
    --dataset_path "../../fundus/_all_images_ORIGINAL" \
    --output_dir "./results"

# FiLM DDPM + L1 + LPIPS（最佳方案）
python train.py --epochs 780 --batch_size 16 --img_size 128 \
    --dataset_path "../../fundus/_all_images_ORIGINAL" \
    --output_dir "./results_film_l1lpips" \
    --film --loss_type l1 --lpips_weight 0.1 --lpips_t 200 \
    --ema_decay 0.9999
```

### 推理采样

```bash
python generate.py \
    --checkpoint "./results_film_l1lpips/models/checkpoint_epoch_000700.pth" \
    --num_images 64 --output_dir "./results_film_l1lpips/generated"
```

### 条件生成（血管骨架引导）

```bash
python train.py --epochs 500 --batch_size 16 --img_size 128 \
    --dataset_path "../../fundus/_all_images_ORIGINAL" \
    --cond_path "./conditions" \
    --output_dir "./results_cond"
```

---

## 📊 训练参数统一 CLI 规范

所有模型的 `train.py` 遵循统一参数接口：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 200 | 总训练轮数 |
| `--batch_size` | int | 16 | 批次大小 |
| `--img_size` | int | 128 | 图像分辨率 |
| `--lr` | float | 0.0002 | 学习率 |
| `--dataset_path` | str | 必填 | 数据集路径 |
| `--output_dir` | str | "./results" | 输出目录 |
| `--resume` | str | None | 断点续训 |

输出目录结构：`{output_dir}/images/`（预览图）、`{output_dir}/models/`（权重）、`{output_dir}/logs/`（日志）。

---

## 📈 后续优化方向

1. **推理后处理**（最容易）：中值滤波去伪影 + ESRGAN/SwinIR 超分 128→256
2. **梯度检查点**：降低显存 30-50%，支持更大 `base_dim`
3. **边缘感知损失**：Sobel/Canny 边缘 L1 损失，减少白色斑点伪影
4. **多尺度 LPIPS**：在 128/64/32 多尺度计算感知损失
5. **多尺度条件注入**：类似 ControlNet 在各分辨率层分别注入条件特征
6. **自监督预训练**：在大量无标注眼底图上预训练 UNet 再微调

---

## 📖 科研报告

完整实验报告位于 [`research-report/REPORT.docx`](research-report/REPORT.docx)（9MB），包含：

- **第 1 章**：摘要与项目背景
- **第 2 章**：数据集与预处理（330 张严重症状眼底彩照）
- **第 3 章**：VAE（第 3 章）与 GAN（第 4 章）实验全记录
- **第 4 章**：扩散模型详细实验（无条件/条件/Palette/FiLM/L1+LPIPS）
- **第 5 章**：结论与展望（含 5.5 学到的经验）

报告由 `generate_docx.py` 脚本生成，修改结构后重新运行即可更新。

---

## 📜 许可证

本项目代码仅供学术研究和教育用途。数据集版权归提供方所有。

---

## 🙏 致谢

感谢指导老师提供的眼底彩照数据集与研究方向的指导。
