# Rectified Flow —— 线性路径 + reflow 拉直

> 方法族：**Rectified Flow（RF）**
> 论文：Liu, Gong, Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data
> with Rectified Flow*, ICLR 2023（arXiv:2209.03003）；规模化见 SD3（Esser et al., ICML 2024）
> 代码参考：`code/external/flow_matching/`（官方库含 rectified flow 示例）；
> 上游 RF 实现：https://github.com/gnobitab/RectifiedFlow

## 与 CFM 的区别

- **路径相同**：都用线性插值 `x_t=(1−t)x0+t·x1`、目标速度 `x1−x0`；
- **区别在 reflow**：RF 会拿第一轮模型生成 `(x0, x1)` 配对，再训第二轮把轨迹拉直，
  从而支持 1–4 步采样。我们目前**没有做 reflow**，只用了线性路径 + 独立耦合。

## 我们的状态：未实现 reflow

- 当前 CFM 的采样是 50/100 步 Euler；reflow 的潜在收益是**交付速度**（少步采样）。
- 成本：需要先跑一轮生成、构造配对数据，再训第二轮；数据量与显存与现有同量级，
  但多一轮训练时间。优先级：中（在时间对齐/去时间条件之后）。
- 注意：reflow 不改变“信息量不足”的问题，只改变轨迹形状；对病理保真预计无帮助。

## 目录

```
RectifiedFlow/
├── README.md
├── models/
└── results/    # 如 results/rf_reflow1/
```
