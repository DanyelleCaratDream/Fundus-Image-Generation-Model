# FundusGen — 小样本眼底彩照生成研究

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-13.0+-76b900.svg)]()

> **在仅 330 张的小样本设定下，系统研究眼底彩照生成模型的失效机理（failure mechanisms）与医学语义评估（evaluation），
> 并将可控的 severe 图生成用于 DR 分级数据扩充。**
> 8GB VRAM 笔记本，VAE → GAN → Diffusion 三代模型完整实验记录 + 传统 ML 基线同口径横评。

---

## 🔬 研究要点

1. **失效机理链条**：小样本眼底生成在三代模型上各有死因——VAE 的像素级 MSE 造成模糊天花板（blur ceiling）；
   GAN 的判别器小样本过拟合；扩散模型的 **MSE 溶解病灶**（lesion dissolution）：暗背景摊薄亮病灶梯度，
   出血保留率 0.254（MSE） vs **1.137**（L1+LPIPS），且在完全不同的架构（Palette）上复现 → 是损失函数的天性而非模型缺陷。
   配套 `--loss_type` 开关使该消融可一键复现。
2. **评估方法论**：两层指标（通用分布层 + 眼底专用层）与六维门控综合评分（six-dimension gated composite score）。
   关键现象：**通用指标看不出医学语义差异**——FID 相近的两个模型，出血保留率相差 4.5 倍。
3. **同口径多方法横评**：11 个方法（5 传统 ML + 6 深度）在同一套指标与评分下对比——文献调研确认这是一个没人做过的空白。
4. **数据量假说（进行中）**：76 张 @256 训练 loss 正常收敛但生成全糊（"拟合 ≠ 生成"）；
   D1/D2 单变量对照实验验证数据量是否为瓶颈。

## 📍 当前状态（2026-10-16）

- **Phase D 进行中**：在老师 DR 分级数据集（758 张四分类）上生成 severe 图 → 老师审核 → 替换数据集自带的 76 张翻转副本 → 扩充数据集训分类器（TSTR）。
- 实验登记表 [EXPERIMENTS.md](generate_project/deep_learning/EXPERIMENTS.md)（DL-EX-001~020）为**全部数字的唯一来源**。
- 论文路线：A《小样本眼底生成：失效机理与评估方法》先行，B（+下游验证）叠加，见 [docs/14-Paper-Spine.md](docs/14-Paper-Spine.md)。

### ⚖️ 分数口径声明（勿混用）

本项目有两套 0-100 分，**含义不同，不可混排**：

| 尺度 | 范围 | 说明 |
|:--|:--|:--|
| 人工分（human score） | 单人目测校准 | film_l1lpips 87.5 / film 85 / base_cj 75 / cond 70 / dcgan 20 / vae 10（`eval/human_scores.csv`，N=6 局限） |
| 六维门控总分（gated composite） | 11 模型可比 | film_l1lpips 72.9 / base_cj 44.2 / cond 44.0 / film 44.0 / vae 25.7 / dcgan 7.0（`eval_data/_scores.json`） |

---

## 📁 项目结构

```
AI_Model_Project_for_Fundus_Color_Images/
├── generate_project/               # 生成方法库（按方法族分区）
│   ├── deep_learning/              #   深度学习（Phase A/B 完成；Phase D 主力进行中）
│   │   ├── EXPERIMENTS.md          #     ⭐ 实验登记表（所有数字的唯一来源）
│   │   ├── Fundus-VAE/             #     VAE 系列（vanilla / beta）
│   │   ├── Fundus-GAN/             #     GAN 系列（DCGAN / WGAN-GP / StyleGAN2-ADA）
│   │   └── Fundus-Diffusion/       #     Diffusion 系列 ⭐（DDPM + FiLM + L1/LPIPS）
│   ├── machine_learning/           #   传统 ML / OpenCV（PCA / GMM / 补丁 / 泊松 / Retinex）
│   ├── pretrained/                 #   预训练路线（调研报告 07：SD-LoRA 主路线）
│   └── transfer_learning/          #   迁移学习（文献归档）
├── classify_project/               # DR 分级分类器（Phase D：TSTR 验证方案）
├── eval/                           # 评估体系（通用层 / 专用层 / 综合评分 / 可视化）
├── eval_data/                      # 评估数据（real 快照 + 各模型生成图 + *_metrics.json）
├── fundus/                         # 数据集（git 排除）：A 域网络图 + 老师 HRSG 数据集 + 卫生工具
├── literature/                     # 文献管理（links + pdf）
├── docs/                           # 规范文档（00 索引 → 15 数据文档）
├── research-report/                # 科研报告与交付物（REPORT_DL / REPORT_ML / evaluation_report）
├── _archive/                       # 归档区（先归档不销毁，ARCHIVE.md 登记）
└── README.md                       # 本文件
```

## 🗺️ 文档地图

| 想了解 | 去哪 |
|:--|:--|
| 文档总索引 | [docs/00-Doc-Index.md](docs/00-Doc-Index.md) |
| 论文主张与口径（写作宪法） | [docs/14-Paper-Spine.md](docs/14-Paper-Spine.md) |
| 数据集与数据卫生 | [docs/15-Data-Documentation.md](docs/15-Data-Documentation.md) |
| 实验数字 | [EXPERIMENTS.md](generate_project/deep_learning/EXPERIMENTS.md)（深度）＋ `machine_learning/docs/06-Records.md`（传统） |
| 当前任务位置 | [research-report/CONTINUE_GUIDE_NEW.md](research-report/CONTINUE_GUIDE_NEW.md) |
| 常用命令与坑 | [docs/08-Work-Guide.md](docs/08-Work-Guide.md) |
| 文献定位（哪些贡献已被做过） | [docs/12-Literature-Survey-and-Positioning.md](docs/12-Literature-Survey-and-Positioning.md) |

## 🚀 快速开始

```bash
pip install torch torchvision numpy Pillow matplotlib lpips pytorch-fid

# 训练（最佳方案：FiLM DDPM + L1 + LPIPS；--loss_type mse 可复现病灶溶解消融）
cd generate_project/deep_learning/Fundus-Diffusion/ddpm
python train.py --epochs 780 --batch_size 16 --img_size 128 \
    --dataset_path "../../../../fundus/_all_images_ORIGINAL" \
    --cond_path ./conditions --film --loss_type l1 \
    --lpips_weight 0.1 --lpips_t 200 --ema_decay 0.9999 \
    --output_dir ./results_film_l1lpips

# 采样（DDIM 50 步；--use_raw 强制原始权重，用于续训 checkpoint 的出图判定）
python generate.py --checkpoint ./results_film_l1lpips/models/final_model.pth \
    --num_images 64 --sampler ddim --sampling_steps 50 \
    --cond_path ./conditions --output_dir ./generated
```

完整命令（训练 / 生成 / 评估 / 评分）见 [docs/08-Work-Guide.md](docs/08-Work-Guide.md)。

## 📖 科研报告

- [`REPORT_DL.docx`](research-report/REPORT_DL.docx) —— 深度学习路线全记录（VAE / GAN / Diffusion，DL-EX-001~014，论文路线 A 的底稿）
- [`REPORT_ML.docx`](research-report/REPORT_ML.docx) —— 传统 ML 路线（PCA / GMM / 补丁基线 + 泊松病变重排 / Retinex 光照交换）
- [`evaluation_report.md`](research-report/evaluation_report.md) —— 评估体系 + 评分标准调研合并版

> 报告为**实验日志体裁**；论文按 claim → evidence 重组，骨架见 [docs/14-Paper-Spine.md](docs/14-Paper-Spine.md)。

---

## 📜 许可证

本项目代码仅供学术研究和教育用途。数据集版权归提供方所有。

## 🙏 致谢

感谢指导老师提供的眼底彩照数据集（A 域网络图 + HRSG 分级数据集）与研究方向的指导。
