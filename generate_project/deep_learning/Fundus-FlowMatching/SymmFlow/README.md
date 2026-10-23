# SymmFlow —— 双向对称流（生成 / 分割 / 分类统一）

> 方法族：**Symmetrical Flow Matching（SymmFlow）**
> 论文：Caetano et al., *Symmetrical Flow Matching: Unified Image Generation,
> Segmentation, and Classification with Score-Based Generative Models*, AAAI 2026
> 代码：`code/external/symmetric-flow/`（已克隆）；
> 权重：https://huggingface.co/ocaetas/SymmFlow（未下载）

## 方法要点

- 同时建模两条相反的流：图像 `x` 从噪声生成，标签 `y` 从数据走向噪声；
- 联合回归 `v=(x−ξ_x, ξ_y−y)`，标签做去量化（加 U(−β,β) 噪声）防止熵塌缩；
- 推理时反向积分可从图像得到标签（分割/分类），正向积分可条件生成；
- 25 步即可生成（CelebA FID 11.9 / COCO 7.0），同时保持分割/分类能力。

## 对我们的意义

- **中期方向**：把“生成 severe 图”和“DR 分级”统一进一个双向模型，
  替代目前分离的 A2 小分类器；生成时可用类别标签做条件，分类时反向积分。
- 改造量较大：需要把我们的 FiLM 条件 UNet 改成同时接收图像与标签两条输入，
  并重新设计损失；数据量（193 张）也可能不够支撑双向训练。
- 短期不排期；先作为 Related Work 与“未来工作”引用。

## 目录

```
SymmFlow/
├── README.md
├── models/     # 若后续下载 HF 权重放这里
└── results/    # 如 results/symmflow_fundus/
```
