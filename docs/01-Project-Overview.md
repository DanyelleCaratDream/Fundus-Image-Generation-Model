# 项目概述

## 项目名称
FundusGen — 眼底彩照生成模型研究项目

## 项目背景
本项目旨在探索并比较多种生成模型在**眼底彩照生成**任务上的表现。项目数据集为 330 张严重症状眼底彩照，由指导老师提供。目标是通过训练生成模型，产生逼真的合成眼底彩照，辅助医学图像分析研究。

## 研究目标
1. 探索四大生成模型家族（VAE、GAN、Diffusion、Flow Matching）在眼底彩照生成上的适用性
2. 为每个模型家族挑选具体模型并训练出最优结果
3. 建立系统化的实验记录与对比体系
4. 生成可供老师检查的结果文件夹
5. 撰写完整科研报告

## 研究路线（老师指定）
从后往前研究：
- VAE（Variational Autoencoder）
- GAN（Generative Adversarial Networks，含 StyleGAN2 等）
- Diffusion Models（DDPM、DDIM、Conditional Diffusion）
- Flow Matching / Mean Flows

每个模型采用"先跑预测（Inference）、再做训练（Training）"的策略。

## 数据集
- **原始图片**：330 张眼底彩照（JPG/PNG 格式）
- **增强图片**：通过水平翻转 + 多角度旋转扩充至 1320 张
- **存储位置**：`D:\fundus\` 及项目内 `fundus/` 目录
- **特点**：全部为严重症状图像，数据量较小，对生成模型的稳定性要求高

## 硬件环境
- **GPU**：NVIDIA GeForce RTX 4060 Laptop GPU（8GB VRAM）
- **CPU**：Intel/AMD 笔记本处理器
- **内存**：16GB+

## 评估标准
- **定量**：FID（Fréchet Inception Distance）、IS（Inception Score）
- **定性**：肉眼观察生成图像的真实感与多样性
