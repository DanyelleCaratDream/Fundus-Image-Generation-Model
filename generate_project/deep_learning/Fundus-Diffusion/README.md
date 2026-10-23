# Fundus-Diffusion —— 自研扩散模型方法族

> 按“框架 → 方法族”组织；每个方法族自含 `upstream/`（外部参考实现）、`models/`、`results/`。
> **预训练/迁移类扩散方法不在这里**，统一放顶层 [`../../pretrained/`](../../pretrained/)，
> 例如 SD2.1+LoRA（B1 线）与 Tiger-Model（文本引导 + ControlNet）。

| 方法族 | 说明 | 状态 |
|:--|:--|:--|
| [`ddpm/`](ddpm/) | 自研 FiLM 条件 UNet + DDPM/DDIM（D3/D4、CFM 训练器的所在地） | ✅ 主线，已产出 D3/D4 |

**约定**：模型权重放各族的 `models/`（按实验名命名）；生成结果放 `results/<实验名>/`；
外部代码放 `upstream/`（不入 git）。
