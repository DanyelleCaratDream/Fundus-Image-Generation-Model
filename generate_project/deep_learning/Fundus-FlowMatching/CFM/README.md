# CFM —— 条件流匹配（Conditional Flow Matching，OT 线性路径）

> 本目录 = `Fundus-FlowMatching/` 下的**具体方法族**（对标 `Fundus-GAN/DCGAN`、
> `Fundus-GAN/WGAN-GP` 的组织方式）。本方法族回答：**用条件流匹配替代 DDPM 的噪声预测，
> 能否在同样的 193 张老师数据上得到更真实的 severe 眼底图。**

## 1. 方法标注（我们到底用的是哪一种 Flow Matching）

| 维度 | 我们的选择 | 说明 |
|:--|:--|:--|
| 框架 | **Conditional Flow Matching (CFM)** | Lipman et al., ICLR 2023（arXiv:2210.02747） |
| 概率路径 | **OT 线性插值路径** `x_t=(1−t)x_0+t·x_1` | 与 Rectified Flow（Liu et al., ICLR 2023）的线性路径同形 |
| 耦合方式 | **独立耦合**（x0、x1 各自采样，不做配对） | **不是** OT-CFM 的 minibatch 最优传输耦合（Tong et al., TMLR 2024） |
| 是否 reflow | **否** | 未做 Rectified Flow 的迭代拉直 |
| 条件注入 | **FiLM**（血管 mask，单通道） | 与 Diffusion 线同一 UNet，仅换训练目标 |
| 采样 | **Euler ODE**，50 NFE（评估）/100 NFE（交付） | 与 DDIM 同 NFE 口径对比 |
| 目标场 | 速度场 `v=x_1−x_0` | 不是噪声 ε、不是 score |

**一句话**：我们做的是 **CFM（线性/OT 路径、独立耦合、FiLM 条件、Euler 采样）**；
不包含 minibatch OT 配对、Schrödinger bridge、stochastic interpolants 的方差保持路径、reflow。
后续若要升级，优先试 OT 耦合（batch≥8 才有意义，我们 batch=1–2 时基本退化）。

## 2. 目录结构

```
CFM/
├── README.md                 # 本文件：方法标注 + 结构 + 结果
├── configs/                  # 三套实验配置（与 results/*/config.json 对应）
│   ├── cfm128.json
│   ├── cfm256.json
│   └── cfm256_lesion.json
├── train/train_cfm.py        # 训练入口（薄封装 → Fundus-Diffusion/ddpm/train.py --objective cfm）
├── infer/
│   ├── sample_cfm.py         # 纯生成（Euler NFE）
│   ├── sdedit_cfm.py         # 自精修（线性路径加噪 + Euler）
│   └── deliver_cfm.py        # 交付批次（SDEdit 0.8，分批 + 断点续跑）
├── tools/
│   ├── make_lesion_weights.py  # 病灶权重图（渗出优先的高精度启发式）
│   ├── lesion_transplant.py    # 病灶块移植（羽化 alpha + LAB 均值匹配）
│   └── tta_hist_match.py       # TTA-Flow 轨迹对齐直方图匹配（推理期，无需重训）
├── data/
│   ├── conditions_combined_256/  # 256 版血管 mask（193 张）
│   └── lesion_weights_256/       # 病灶权重图（193 张）
├── results/
│   ├── cfm128/               # 320 轮，loss 0.90→0.12–0.14
│   ├── cfm256/               # 320 轮，loss 0.100（作者 97/100）
│   └── cfm256_lesion/        # 病灶加权续训（λ=8，150 轮，进行中）
└── judgment/                 # 评估/对照图（同槽对比、移植对照、交付总览）
```

**为什么训练器不在本目录**：CFM 与 Diffusion 共用同一套 FiLM 条件 UNet、数据管线与
温度/EMA 逻辑；复制一份会分叉。`train/train_cfm.py` 是指定 `--objective cfm` 的入口，
真正的实现见 `../../Fundus-Diffusion/ddpm/train.py`（`cfm_path` / `cfm_predict_x1` / `euler_sample`）。

## 3. 三套实验

| 实验 | 初始化 | 数据/条件 | 轮数 | 结果 |
|:--|:--|:--|--:|:--|
| **cfm128** | film_l1lpips final（A 域最佳） | combined_sq128 / mask128 | 320（780→1100） | loss 0.90→0.12–0.14；8/8 结构完整、色彩比 D3 近真实，血管偏软 |
| **cfm256** | cfm128 final | combined_sq256 / mask256 | 320（1100→1420） | loss 0.100；**作者目检 97/100**；锐度 0.00295≈D3 0.00313 |
| **cfm256_lesion** | cfm256 final | 同上 + 病灶权重 λ=8 | 150（1420→1570） | 进行中；目标提升出血/渗出保留 |
| **cfm128_align**（消融） | film_l1lpips final | 同 cfm128，但时间轴反转对齐 DDPM | 320（780→1100） | 待跑（GPU 空出后）；验证“时间语义对齐”是否加速/稳定迁移 |

> **时间轴问题（2026-10-23 发现）**：DDPM 的 t=0 是干净图、999 是噪声；CFM 的 t=0 是噪声。
> 原实现把 CFM 的 t 直接映射到 0..999，等于让模型“倒着学”时间条件（首轮 loss 0.904≈预测零速度）。
> `--cfm_time_align` 改为 `idx=(1−t)·T`；采样端（generate/sdedit/deliver）会从 checkpoint
> 的 `args.cfm_time_align` 自动读取同一约定。

## 4. 结果摘要（B 域三模型，60 张/模型，severe 参照 72 张）

| 指标 | D3 | CFM@128 | **CFM@256** |
|:--|--:|--:|--:|
| FID ↓ | 222.69 | 178.87 | **121.84** |
| KID ↓ | 0.1287 | 0.1634 | **0.0902** |
| 颜色均值距离 ↓ | 0.2996 | 0.0782 | **0.0345** |
| BRISQUE ↓（真实 11.02） | 42.83 | 38.00 | **13.21** |
| 出血保留率 ↑ | **0.754** | 0.320 | 0.204 |

**结论**：CFM@256 在真实性/分布上全面领先；D3 在病灶保留上更强——真实性与病理保真的取舍。
自精修在 256 上有效（SDEdit 0.8：锐度 +9.5%、色彩不变），在 128 上几乎无增益。

## 5. 运行

```bash
# 训练（示例：cfm256_lesion；实际路径以 configs/*.json 为准）
python train/train_cfm.py --preset cfm256_lesion

# 生成 8 张判定图
python infer/sample_cfm.py --ckpt results/cfm256/models/final_model.pth \
    --num 8 --seed 42 --steps 50 --out judgment/cfm256_8

# 自精修
python infer/sdedit_cfm.py --ckpt results/cfm256/models/final_model.pth \
    --base judgment/cfm256_8/singles --t_start 0.8 --out judgment/cfm256_s08

# 交付批次（60 张，SDEdit 0.8）
python infer/deliver_cfm.py --base <60 张底图目录> --out judgment/delivery_s08
```

## 6. 纪律

- 采样 NFE 与 DDIM 对齐（50 评估 / 100 交付），不许拿 1000 步打 50 步；
- 生成器只用 train，不碰 val/test；
- 验收次序：**专家目检 > 分类器指标**；
- 温度 ≤70°C（训练器内置每 25 步守护）。
