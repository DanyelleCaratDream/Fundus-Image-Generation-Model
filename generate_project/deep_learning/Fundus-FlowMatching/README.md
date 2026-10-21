# Fundus-FlowMatching —— 条件流匹配（Conditional Flow Matching, CFM）方法族

> 建立于 2026-10-21。老师推荐重点研究**条件流匹配方法族**（"直接看这个方法族"），本目录承载：
> 论文调研、外部实现、我们的 CFM 训练/采样、结果与判定。
> 与 `Fundus-Diffusion/` 的关系：CFM 是**同一套 FiLM 条件 UNet 的另一种训练目标**，
> 现成代码在 `../Fundus-Diffusion/ddpm/train.py`（`--objective cfm`），本目录负责把它独立成线。

## 1. 方法族一句话

CFM 学一个**速度场** `v_θ(x_t, t)`，把噪声沿一条（理想上是直线的）概率路径推到数据；
训练是**回归**（simulation-free，不需要模拟 ODE），采样是**积分 ODE**（Euler/RK）。
它是扩散模型的推广：DDPM 的 probability-flow ODE 是其中一条特定路径的特例。

OT 线性路径（我们已实现的形式）：

```
x_0 ~ N(0, I)                         # 噪声
x_1 ~ q_data                          # 数据
x_t = (1 - t) · x_0 + t · x_1         # t ∈ [0,1]
v   = x_1 - x_0                       # 目标速度（沿直线，恒定）
L   = E || v_θ(x_t, t) - v ||²        # 回归损失
采样：dx/dt = v_θ(x_t, t)，t: 0 → 1，Euler 积分
```

与扩散的关键差别：
| | 扩散（ε-pred / DDPM-DDIM） | 条件流匹配（OT-CFM） |
|:--|:--|:--|
| 预测目标 | 噪声 ε | 速度 v = x1 − x0 |
| 路径 | 方差保持/爆炸的弯曲路径 | 线性插值，可被 reflow 拉直 |
| 训练 | 加噪 + 去噪回归 | 插值 + 速度回归（同样 simulation-free） |
| 采样 | DDIM/DDPM，N 步 | Euler/RK，N 步（NFE 口径一致可直接对比） |
| 条件 | cross-attn / FiLM / concat | 同左（条件只进网络，不改路径） |

## 2. 我们的现状（2026-10-21）

- **代码已就位、从未训练**：`train.py` 实现了 OT 线性路径 `cfm_path()`（行 385）、
  由速度反推 x1 `cfm_predict_x1()`（行 400，供 LPIPS/颜色正则）、Euler 采样 `euler_sample()`
  （行 406）；`generate.py` 有 `--sampler auto|euler` 与 `--objective` 覆盖。
  证据：`grep -n "cfm" train.py generate.py`。
- **从未跑过 CFM**：EXPERIMENTS.md 全文无 cfm/flow-matching 实验记录；D3/D4 config 均为 ddpm。
- **下一步（F2，作者已排期）**：除 `--objective cfm` 外一切照抄 D3 config
  （`resume=film_l1lpips/final_model.pth`、`combined_sq128`、`conditions_combined_128`、
  L1+LPIPS 0.1+color 0.01、bd128、batch2、1100ep、同种子），与 D3_ema 同槽对决。

## 3. 目录约定

```
Fundus-FlowMatching/
├── README.md                 # 本文件：方法族总览 + 现状
├── papers/
│   ├── reading-list.md       # 顶会论文表（venue/年份/链接/贡献/与本项目相关性）
│   └── notes/                # 逐篇精读笔记
├── code/
│   ├── README.md             # 外部库清单 + 我们实现的指针
│   └── external/             # 克隆的外部仓库（不入 git，见 .gitignore 约定）
└── results/                  # CFM 训练产物（权重不入 git）
```

## 4. 纪律

- 数字与判定写 EXPERIMENTS.md（DL-EX-026 起编号），本目录只放方法与素材；
- 采样步数口径：CFM Euler 与 DDIM 用**相同 NFE**（默认 50），不许 1000 步打 50 步；
- 交付门是**专家过眼**，不是指标（见 memory: expert-acceptance-first）。
