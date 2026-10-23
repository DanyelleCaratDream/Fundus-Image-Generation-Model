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

## 运行前需要

1. 下载权重：[HF FangDai/Tiger-Model](https://huggingface.co/FangDai/Tiger-Model)
   或 README 里的两个 Google Drive 链接（Coarse / Fine 两阶段）；
2. 依赖：`diffusers`、`accelerate`、ControlNet；
3. 数据格式与提示词见 `upstream/tiger-model/README.md`。

## 与主线的优先级

低于 CFM 的时间对齐/去时间条件/病理保真实验；属于“方法对照/备选路线”，
在 CFM 主线稳定后再排期。
