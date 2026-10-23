# OT-CFM —— 最优传输耦合的条件流匹配

> 方法族：**OT-CFM**（Optimal Transport Conditional Flow Matching）
> 论文：Tong et al., *Improving and Generalizing Flow-Based Generative Models with
> Minibatch Optimal Transport*, TMLR 2024（arXiv:2302.00482）
> 代码参考：`code/torchcfm/`（`ExactOptimalTransportConditionalFlowMatcher`）

## 与 CFM 的区别

CFM 用**独立耦合**：每个 batch 里 `(x0, x1)` 随机配对。
OT-CFM 先用 minibatch 内的最优传输给出配对方案，再沿线性路径回归速度——
轨迹更直、训练更稳、推理步数更少。

## 我们的状态：未实现（优先级低）

- 关键限制在 **batch size**：batch=1–2 时 minibatch OT 几乎没有可配对的组合，
  退化为独立耦合；我们 256 分辨率下 batch=1、128 下 batch=2，显存限制了这条路。
- 若要做，至少需要 batch≥8（128 分辨率、bd64 或更小模型），或换更大显存的机器。
- 它是论文里“方法族对比”的一个候选，但不是当前主线。

## 目录

```
OT-CFM/
├── README.md
├── models/     # 本族的模型权重（目前为空）
└── results/    # 按实验名建子目录，如 results/otcfm128/
```
