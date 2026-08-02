# 执行路线图

> 更新日期：2026-08-01 ｜ 已批准计划：C:\Users\Yelle\.claude\plans\ai-fundus-sparkling-feather.md

---

## 总体路线

```
Phase A 评估体系（老师 Bug 1 第一半，最高优先）
   ↓
Phase C1 经典增广（老师 Bug 2，便宜可靠）
   ↓
Phase B 指标调研（Bug 1 第二半，可并行）
   ↓
Phase C3 Stable Diffusion（可选探索，靠后）
   ↓
Phase D 下游分类器验证（终极目标）
```

---

## Phase A：评估指标体系 + 历史模型重算 🔴 最高优先

| 步骤 | 任务 | 说明 |
|:--|:--|:--|
| A1 | 生成评估图 | 6 个有 checkpoint 的模型各生成 300 张单张图到 `eval_data/{model}/singles/`；真实基准集 = 330 张 ORIGINAL resize 128 |
| A2 | 通用层指标 | `metrics_common.py`：FID/IS/KID/P&R/Density/Coverage/1-NN/MS-SSIM/BRISQUE |
| A3 | 专用层指标 | `metrics_fundus.py`：Vessel Dice/血管分布/病灶P-R/面积分布/颜色距离/C2ST/记忆检测 |
| A4 | 产出报告 | `research-report/evaluation_report.md`：6 模型 × 全部指标对比表 |

**可评估模型**（有 checkpoint）：FiLM+L1+LPIPS（最佳）、FiLM MSE、条件扩散、基础 DDPM、DCGAN、VAE Large 800。
**跳过**：WGAN-GP / StyleGAN2 / VAE 1200（权重已丢失，用户确认）。

**⚠️ 已知障碍（下个会话先处理）：**
1. 条件扩散 / 基础 DDPM 的 checkpoint 加载出现 size mismatch，架构参数需确认（诊断：cond 用 bd=64 ssn=False 接近但 miss=88/unexp=90，可能 attn_layers/cond_channels/EMA 格式问题）
2. DCGAN / VAE 的 generate.py 需给 `torch.load` 加 `weights_only=False`（第 64 / 68 行）

## Phase B：评分标准调研（Bug 1 第二半）✅ 已完成 2026-08-02

文献调研医学图像生成评估 / 眼底图专用指标。考察：领域适配 FID（眼底特征提取器）、结构保持、下游效用（TSTR/TRTR）、小样本友好指标。结论已并入 `research-report/evaluation_report.md` 第二部分。核心：保留两层指标 + TSTR/TRTR 为终极金标准 + RETFound-FD 可选探索。

## Phase C：新方法探索（Bug 2）

| 子项 | 内容 | 状态 |
|:--|:--|:--|
| C1 经典增广 | OpenCV 弹性形变/几何/轻微颜色扰动，从 330 张扩 3~5 倍；严格避免重 ColorJitter（医学颜色有生理意义） | 🔴 必做 |
| C2 传统 ML | PCA/GMM/补丁合成快速实验，预期否决（难以产出真实眼底纹理），确证则作为否定性结论写入报告 | 🟡 1 次实验 |
| C3 Stable Diffusion | 先调研可行性（8GB VRAM，LoRA 512 输出 vs 128 评估体系对接），可行再跑 | 🟢 可选 |

## Phase D：下游分类器验证（终极目标）

1. 下载公开 DR 数据集（优先 APTOS 2019，需 Kaggle 账号；或 EyePACS 子集），老师后续提供真实数据
2. ResNet/EfficientNet 小模型做 DR 0-4 分级
3. TSTR vs TRTR：真实图 vs 真实图+合成重度图，重点看 4 级类 Recall/F1/平衡准确率
4. 终极验收：合成重度图显著提升分类器对重度类的性能

---

## 当前状态（2026-08-01）

| 任务 | 状态 |
|------|:--:|
| 工程前置文档完善 | ✅ 完成 |
| CONTINUE_GUIDE 迁移 + 更新 | ✅ 完成 |
| Phase A：评估图生成 | ⏳ 未开始（依赖解决上述 2 个障碍） |
| Phase B/C/D | ⏳ 未开始 |

> 详细执行命令见 [08-Work-Guide.md](08-Work-Guide.md)；当前进度与下一步见 `research-report/CONTINUE_GUIDE_NEW.md`。
