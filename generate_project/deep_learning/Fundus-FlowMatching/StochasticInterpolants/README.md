# Stochastic Interpolants —— 随机插值统一框架

> 方法族：**Stochastic Interpolants（SI）**
> 论文：Albergo & Vanden-Eijnden, *Building Normalizing Flows with Stochastic Interpolants*,
> ICLR 2023（arXiv:2209.15571）；扩展版 Albergo, Boffi, Vanden-Eijnden, JMLR 2025
> 代码参考：`code/external/flow_matching/`（含 stochastic interpolant 路径）；
> 上游：https://github.com/malbergo/stochastic-interpolants

## 与 CFM 的区别

SI 给出更一般的插值 `x_t = α(t)x_1 + β(t)x_0 + γ(t)z`，通过选择 `α,β,γ` 覆盖：
- 线性/OT 路径（我们的形式，γ=0）；
- 方差保持（VP）路径（与 DDPM 的概率流 ODE 对应）；
- 带随机项的插值（可用 SDE 采样，也可取概率流 ODE）。

换句话说：**CFM 是我们现在用的一个特例**；SI 是包含它和扩散的统一框架。

## 我们的状态：参考，未实现

- 实现成本中等：只需替换路径函数与目标速度的构造，训练循环可复用。
- 研究点：VP 路径 vs 线性路径，在 72 张 B 域数据上谁更稳、细节更好——
  这正好能解释“为什么 DDPM→CFM 迁移时时间轴会反”（两条路径的时间参数化不同）。
- 优先级：低–中；更适合作为论文的方法族讨论，而不是交付主线。

## 目录

```
StochasticInterpolants/
├── README.md
├── models/
└── results/    # 如 results/si_vp128/
```
