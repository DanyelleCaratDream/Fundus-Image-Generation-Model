# TextGuidedDiffusion —— 文本引导扩散（方法族）

> 本目录是 `Fundus-Diffusion/` 下的**具体方法族**之一（另一个是 `ddpm/`）。
> 当前承载的是 **Tiger-Model**（罕见甲状腺癌，Nature Communications 2025）的参考实现：
> 用文本描述 + ControlNet 条件做罕见类图像增广。

## 为什么会在这里

它不是 Flow Matching，而是**扩散模型**（Stable Diffusion + ControlNet），
所以按框架归到 `Fundus-Diffusion/` 下，与自研 `ddpm/` 并列。

## 状态：📌 代码已下载，**未跑**

作者 2026-10-23 明确要求：**后续提醒可以跑一跑这个方法**。

它可以用来回答：
- 文本条件（如“severe DR with hemorrhages and hard exudates”）能否比 mask 条件更好地注入病理语义？
- 在 72 张 severe 上，ControlNet 式结构条件 + 文本引导是否比纯 LoRA 稳？
- 罕见类增广的下游分类收益（Tiger-Model 原文的核心证据）。

## 目录

```
TextGuidedDiffusion/
├── README.md
├── upstream/tiger-model/     # 官方代码（zip 下载，未含权重；不入 git）
├── models/                   # 若后续下载权重放这里（HF FangDai/Tiger-Model / Google Drive）
└── results/                  # 按实验名建子目录
```

## 与主线的优先级

低于 CFM 的时间对齐/去时间条件/病理保真实验；属于“方法对照/备选路线”，
在 CFM 主线稳定后再排期。

## 它和我们的 ddpm 到底差在哪

Tiger-Model **不是一个新的扩散目标**（不是新损失、也不是新采样器），
而是“预训练潜在扩散 + 文本条件 + ControlNet + 两阶段微调”的一套**流程**。
和自研 ddpm 的差别：

| 维度 | 我们的 ddpm | Tiger-Model |
|:--|:--|:--|
| 生成空间 | **像素空间**（128/256） | **潜空间**（Stable Diffusion VAE） |
| 条件 | **血管 mask**，FiLM 注入 | **文本提示 + ControlNet**（结构条件） |
| 初始化 | 从零，或 film_l1lpips 微调 | **预训练 SD + ControlNet** |
| 训练阶段 | 单阶段 | **两阶段**：Tiger-Coarse（结构）→ Tiger-Fine（细节） |
| 数据领域 | 眼底彩照 | 病理切片（甲状腺癌亚型） |
| 目的 | 扩充 severe 训练集 | 罕见亚型增广，提升分类准确率 |

## 我们能借鉴什么

1. **文本条件作为语义控制**：比如“severe DR with hemorrhages and hard exudates”，
   比单一的 mask 条件更直接地表达“病灶语义”。代价是要给我们的 UNet 加 CLIP 文本编码器
   与 cross-attention（中等改造量）。
2. **两阶段 coarse→fine**：先用低分辨率/粗条件定结构，再高分辨率补细节。
   我们已经有“128→256”的雏形，但 Tiger 是显式的两阶段目标，可以借鉴其训练顺序。
3. **数据筛选/质量控制**：Tiger 用分类器筛生成图，对应我们计划中的 A2 小分类器。
4. **罕见类增广的叙事**：severe 稀缺与罕见癌亚型稀缺同构，适合写进 Related Work / Discussion。

## 要不要“从头训这个方法”？

**不建议。** 理由：

- Tiger 的效果主要来自**预训练先验 + ControlNet + 文本语义**，不是扩散目标本身；
  从零训练等于把这些优势全部丢掉，退化成“又一个 DDPM”，看不出方法价值；
- 它依赖 Stable Diffusion 权重与 ControlNet，从零训的成本远超我们 8GB 单卡的能力；
- 我们只有 193 张数据，不足以支撑文本-图像对齐训练。

如果真要验证，正确的方式是：
① 用其发布的权重做迁移/推理（HF 或 Google Drive），观察在眼底上的零样本表现；
② 或者在我们自己的 ddpm/CFM 上实现一个**轻量文本条件分支**（CLIP + cross-attn），
   与 mask 条件做 A/B——这才是对主线有信息量的实验。

## 运行前需要

1. 下载权重：[HF FangDai/Tiger-Model](https://huggingface.co/FangDai/Tiger-Model)
   或 README 里的两个 Google Drive 链接（Coarse / Fine 两阶段）；
2. 依赖：`diffusers`、`accelerate`、ControlNet；
3. 数据格式与提示词见 `upstream/tiger-model/README.md`。
