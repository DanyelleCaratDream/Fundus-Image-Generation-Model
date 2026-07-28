# 执行计划

## 总体优先级

```
第1梯队 [立即开始]:  VAE + DCGAN/WGAN-GP 重构与统一
第2梯队 [重点攻关]:  StyleGAN2-ADA + DDPM/DDIM 实现与训练
第3梯队 [有余力做]:  Conditional Diffusion + Flow Matching
```

## Step 1: 建立规范文件体系

| 任务 | 产出 | 预计时间 |
|------|------|---------|
| 创建 docs/ 目录 | 7份规范文档 | ~30 min |
| 创建各项目 STANDARDS.md | 4 份项目规范 | ~15 min |
| 清理旧项目文件 | 整理目录结构 | ~15 min |

## Step 2: 重构 Fundus-VAE

| 任务 | 产出 | 预计时间 |
|------|------|---------|
| 编写统一 train.py | 适配 CLI 规范的 VAE 训练脚本 | ~60 min |
| 编写统一 generate.py | 适配 CLI 规范的生成脚本 | ~30 min |
| 跑预测（预训练权重） | 确认模型能生成图片 | ~15 min |
| 训练 vanilla VAE | 训练结果 + 模型档案 | ~60 min |
| 训练 beta-VAE | 训练结果 + 模型档案 | ~60 min |

## Step 3: 重构 Fundus-GAN

| 任务 | 产出 | 预计时间 |
|------|------|---------|
| DCGAN - 统一 train.py | 适配 CLI 规范 | ~30 min |
| WGAN-GP - 统一 train.py | 适配 CLI 规范 | ~30 min |
| StyleGAN2-ADA - 统一训练 | 适配 CLI 包装器 | ~60 min |
| 训练 DCGAN | 训练结果 | ~2-8 hr |
| 训练 WGAN-GP | 训练结果 | ~2-8 hr |
| 训练 StyleGAN2-ADA | 训练结果 | ~4-12 hr |

## Step 4: 实现 Fundus-Diffusion

| 任务 | 产出 | 预计时间 |
|------|------|---------|
| 实现 DDPM 模型 | DDPM 模型代码 | ~120 min |
| 实现 DDIM 采样 | DDIM 加速采样 | ~60 min |
| 实现统一 train.py | 训练脚本 | ~60 min |
| 实现统一 generate.py | 生成脚本 | ~30 min |
| 训练 DDPM | 训练结果 | ~4-12 hr |
| 训练 DDIM | 训练结果（基于 DDPM） | ~2-4 hr |

## Step 5: 实现 Fundus-Flow（可选）

| 任务 | 产出 | 预计时间 |
|------|------|---------|
| 实现 Flow Matching | 模型代码 + 训练脚本 | ~180 min |
| 训练 Flow Matching | 训练结果 | ~4-12 hr |

## Step 6: 对比总结 + 科研报告

| 任务 | 产出 | 预计时间 |
|------|------|---------|
| 汇总所有实验结果 | 对比表格 + 生成样例图集 | ~60 min |
| 撰写科研报告 | report.md | ~120 min |

## 时间预估总览

| 阶段 | 预估总时间 |
|------|-----------|
| Step 1-2 (规范 + VAE) | ~4-5 hr |
| Step 3 (GAN) | ~8-20 hr |
| Step 4 (Diffusion) | ~8-20 hr |
| Step 5 (Flow, 可选) | ~8-16 hr |
| Step 6 (报告) | ~3 hr |
| **合计** | **~30-60 hr** |

注：训练时间取决于 GPU 算力和参数设置，以上为 RTX 4060 8GB 的估算。
