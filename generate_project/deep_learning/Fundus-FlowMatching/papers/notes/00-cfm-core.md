# CFM 核心笔记（奠基三篇 + 我们的实现对照）

> 2026-10-21 初读。公式与结论以 Lipman et al. (ICLR 2023) 为主线，
> 与 `../Fundus-Diffusion/ddpm/train.py` 的实现逐项对照。

## 1. 问题设定

要学一个连续归一化流（CNF）：ODE `dx/dt = v_θ(x_t, t)`，把先验 `p_0 = N(0, I)`
推到数据分布 `p_1 = q_data`。直接回归边际速度场 `u_t(x)` 不可行（不知道 `p_t`）。
CFM 的技巧：**构造条件路径** `p_t(x | x_1)`，回归其条件速度 `u_t(x | x_1)`，
并证明两者梯度相同。

## 2. 条件路径与损失

给定数据点 `x_1`，选一条从噪声到数据的路径。两种常用：

**（a）Gaussian 条件路径（Lipman 原文默认）**
```
x_t = α_t · x_1 + σ_t · ε,   ε ~ N(0, I)
u_t = α'_t · x_1 + σ'_t · ε
```
方差保持（VP）取 `α_t = cos(πt/2), σ_t = sin(πt/2)`。

**（b）OT 线性路径（我们的实现）**
```
x_t = (1 - t) · x_0 + t · x_1,   x_0 ~ N(0, I)
v   = x_1 - x_0                  # 恒定速度，路径是直线
```
对应 `α_t = t, σ_t = 1 - t`。这就是 `train.py:385 cfm_path()`：
```python
x_t = (1.0 - t) * noise + t * x1
v_target = x1 - noise
```

**损失（两种路径同形）**
```
L_CFM(θ) = E_{t~U[0,1], x_1~q, x_0~p_0} || v_θ(x_t, t) - v_target ||²
```
关键定理（Lipman 2023, Thm 2）：`∇_θ L_CFM = ∇_θ L_FM`，所以只回归条件速度即可。

## 3. 采样

从 `x_0 ~ N(0, I)` 出发，Euler 积分 `x ← x + Δt · v_θ(x, t)`，`t: 0 → 1`。
步数 = NFE。我们的 `train.py:406 euler_sample()` 即此，且注释明确：
"评估时与 DDIM 使用相同步数（默认 50），不要拿 1000 步打 50 步。"

## 4. 与扩散的关系（写论文时的关键段落）

- DDPM 的 **probability-flow ODE** 是 CFM 在一个特定（VP）路径下的特例；
- 扩散预测 ε，等价于在特定路径上预测速度的线性变换；
- CFM 的优势不在"表达力"，而在**路径可选择**：选直线路径 → 少步采样、训练更稳；
- 因此"CFM vs Diffusion"不是两个物种，而是**同一族目标在不同路径/参数化下的选择**——
  这正好契合我们论文"失效机理链"的叙述：换目标不等于换信息量。

## 5. 与本项目实现的逐项对照

| 论文概念 | 我们的代码 | 位置 |
|:--|:--|:--|
| OT 线性路径 `x_t, v` | `cfm_path(x1, t_cont)` | `train.py:385` |
| 由 `v` 反推 `x1`（供 LPIPS/颜色正则） | `cfm_predict_x1(x_t, t, v_pred)` | `train.py:400` |
| Euler 采样 | `euler_sample(model, shape, steps, cond)` | `train.py:406` |
| 连续 t → 离散 embedding 下标 | `t_cont_to_index()` | `train.py:380` |
| 训练目标开关 | `--objective {ddpm,cfm}`，写入 config.json | `train.py:107` |
| 采样器自动选择 | `--sampler auto`（cfm→euler） | `generate.py:63` |
| 条件注入 | FiLM（mask 条件），与 objective 无关 | `train.py` ResBlock |

**结论：F2 实验零代码改动**——只需 `--objective cfm`，其余照抄 D3 config。

## 6. 待精读的问题（下一步）

1. **OT 耦合**（Tong 2024）在 193 张小数据上是否值得？——minibatch OT 需要 batch ≥ 2，
   我们 batch=2 时耦合空间极小，可能退化为独立耦合；需读原文的 batch 敏感性。
2. **直线路径 + L1/LPIPS 损失**是否兼容？我们 CFM 分支已用 `cfm_predict_x1` 反推 x1
   再算 LPIPS，需确认这个近似在 t 接近 1 时是否稳定。
3. **t 采样分布**：SD3 用 logit-normal 偏感知尺度；我们目前是 `torch.rand` 均匀。
   小数据下是否有增益需要实验。
4. **reflow**（Liu 2023）是否值得做第二轮：把 1 步/2 步采样变成可能，对交付速度有意义。
