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

## 2. 我们的现状（2026-10-23 更新：CFM 已训练并胜出）

**CFM 不是"待做"，而是已经跑完两条线**——只是训练器住在
`../Fundus-Diffusion/ddpm/train.py`（共用 FiLM 条件 UNet，`--objective cfm` 切换），
结果已归位到本目录 `results/`：

| 实验 | 配置 | 结果 | 目录 |
|:--|:--|:--|:--|
| **F2（CFM@128）** | 从 film_l1lpips 初始化，其余复刻 D3；320 轮 | loss 0.90→0.12–0.14；8/8 结构完整、色彩接近真实，血管偏软 | `results/cfm128/` |
| **F3（CFM@256）** | 从 CFM@128 初始化，combined_sq256 + 256 mask；320 轮 | loss 0.100；**作者目检 97/100**；锐度 0.00295≈D3 0.00313、色彩距离最小 | `results/cfm256/` |

B 域三模型同协议评估（60 张/模型，severe 参照 72 张）：

| 指标 | D3 | CFM@128 | **CFM@256** |
|:--|--:|--:|--:|
| FID ↓ | 222.69 | 178.87 | **121.84** |
| KID ↓ | 0.1287 | 0.1634 | **0.0902** |
| 颜色距离 ↓ | 0.2996 | 0.0782 | **0.0345** |
| BRISQUE ↓（真实 11.02） | 42.83 | 38.00 | **13.21** |
| 出血保留率 ↑ | **0.754** | 0.320 | 0.204 |

**结论**：CFM@256 在真实性/分布上全面领先，D3 在病灶保留上更强——真实性与病理保真的取舍。
下一步（进行中）：病灶区域加权训练、病灶移植。

### 2.1 代码入口（无需复制）

`train.py` 实现 OT 线性路径 `cfm_path()`、`cfm_predict_x1()`、Euler `euler_sample()`；
`generate.py` 的 `--sampler auto` 对 CFM checkpoint 自动选 Euler（2026-10-22 修复分派 bug 后）。
训练命令见 `code/README.md`；新实验输出请写到本目录 `results/<exp>/`。

## 3. 目录约定

对标 `Fundus-GAN/DCGAN`、`Fundus-GAN/WGAN-GP` 的组织方式：**方法族目录下自含训练、推理、
中间产物与结果**。

```
Fundus-FlowMatching/
├── README.md                 # 方法族总览
├── papers/                   # 文献调研（reading-list + 精读笔记）
├── code/external/            # 上游库克隆（不入 git）
│
├── CFM/                      # 方法族 1：条件流匹配（线性 OT 路径，独立耦合）★ 主线
│   ├── README.md
│   ├── upstream/tta-flow/    # TTA-Flow 官方代码（CFM 应用，不入 git）
│   ├── configs/ train/ infer/ tools/ data/ judgment/
│   ├── models/               # 本族全部模型权重（cfm128/cfm256/...）
│   └── results/              # 按实验名：cfm128/、cfm256/、cfm256_lesion/
│
├── OT-CFM/                   # 方法族 2：minibatch 最优传输耦合（未实现，低优先）
├── RectifiedFlow/            # 方法族 3：线性路径 + reflow（未实现，中优先）
├── StochasticInterpolants/   # 方法族 4：随机插值统一框架（参考）
└── SymmFlow/                # 方法族 5：双向流（参考/中期）
```

**每个方法族的统一约定**（作者 2026-10-23 定）：

- `models/` 放该族**所有实验的模型权重**，用实验名命名（如 `cfm128_final_model.pth`）；
- `results/<实验名>/` 放该实验的生成结果、日志、config——即“族文件夹 → results → 实验名”的孙子目录；
- 同一族只是换了训练目标/条件/技巧，不另开族文件夹，都在族内用不同实验名区分；
- 上游代码放 `code/external/`，不复制进方法族，避免分叉。

## 4. 纪律

- 数字与判定写 EXPERIMENTS.md（DL-EX-026 起编号），本目录只放方法与素材；
- 采样步数口径：CFM Euler 与 DDIM 用**相同 NFE**（默认 50），不许 1000 步打 50 步；
- 交付门是**专家过眼**，不是指标（见 memory: expert-acceptance-first）。
