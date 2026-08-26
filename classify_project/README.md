# classify/ —— DR 分级分类模型（Phase D）

> **方案 1**：与生成项目同仓库（保留记忆/文档/数据共享），独立目录做分类。

## 任务定位
用生成数据（`eval_data/` 各模型）+ 老师的分级数据集（`fundus/HRSG_4_758_v0.3.4`）平衡类别分布，训练 DR 分级分类器，验证"合成 severe 图对分类器的价值"。

## 老师数据集（2026-08-26 到位，Phase D 前提已解决）
- 位置：`fundus/HRSG_4_758_v0.3.4/`（4 类 × 3 split，745 张，带分级标签，全 jpg）
- 类别：normal / mild / moderate / severe（即 KW 分级口径，本项目生成 severe 最重图）
- **实测计数**：
  | | normal | mild | moderate | severe |
  |:--|--:|--:|--:|--:|
  | train | 120 | 140 | 140 | **152** |
  | val | 34 | 40 | 40 | **21** |
  | test | 18 | 20 | 20 | **13** |
- ⚠️ 注意：train 里 severe 已是最多（152>140）；"severe 少"只在 val/test 成立
- ⚠️ severe 类分辨率系统性偏低（238-800px 居多，其他类普遍 1600-3456px）
- ⚠️ 文件名已有 `_aug_1` 后缀（数据集自带一次几何增广）

## 已锁定决策（用户 2026-08-26 确认）
- **补图范围**：train + val，**test 保留真实**（TSTR/TRTR 严谨性，防合成图刷 test 分）
- **平衡目标**：severe 补齐到 mild/moderate 数 → train +0（152 已够）、**val +19**、test +0
- **生成图分辨率**：重新生成 **512×512 原生图**（poisson/retinex 原生工作 512，128 只是评估输出）
- **生成器待定**：先试所有生成器（含未来预训练模型）再决定

## 生成器对比（TSTR/TRTR 脚手架，待建）
- 目标：回答"哪个生成器对 severe 增广最有用"
- 协议：每个生成器跑一轮"真实 severe + 该生成器合成 severe → 真实 test 评估"
- 对照：真实-only 训练（不掺合成）作基线
- 输出：severe 类 Recall / F1 / 平衡准确率 / 混淆矩阵 对比表
- 分类器设计（建议）：ResNet18 ImageNet 预训练，224 输入，4 类分级
- 候选：poisson（80.8）/ retinex（72.5）/ film_l1lpips（45.2）/ vae / cond + 基线（pca/gmm/patch 反面对照）+ 未来预训练模型（SD LoRA 等）

## 待办
- [ ] 预训练生成模型调研（`machine_learning/report/07-pretrained-gen-models.md`）→ 定可试的预训练方案
- [ ] 生成 512×512 合成 severe 图（各候选生成器）
- [ ] TSTR/TRTR 脚手架（分类器 + 数据管线 + 评估协议）
- [ ] 跑对比 → 决定用哪个生成器 → 补图（train+val，val +19）
- [ ] 验收：合成重度图显著提升对应类 Recall/F1/平衡准确率

## 数据/工具入口
- 真实图（训练生成器用）：根目录 `fundus/_all_images_ORIGINAL/`（330 张，网络下载，与老师数据集无关）
- 老师分级数据集：`fundus/HRSG_4_758_v0.3.4/`（train/val/test × 4 类）
- 生成图：根目录 `eval_data/{model}/singles/`（旧方法 300 张；新方法 60 张，128px 评估口径）
- 血管骨架：`generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/`（330 张）
- 评分体系：根目录 `eval/`（score_scheme 可评估生成质量；分类价值由 TSTR/TRTR 判）
