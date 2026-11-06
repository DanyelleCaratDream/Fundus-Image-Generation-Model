# 实验复现计划：TTA-Flow（OCT 跨设备测试时自适应）

> 生成日期：2026-10-24
> 论文文件：`uploaded_document.md`（用户上传，由 WisDoc 解析）
> 官方仓库：https://github.com/Veit21/tta-flow（commit `c2f1190`，已本地克隆核查）

## 1. 论文摘要

- **标题**：Test-Time Adaptation in Optical Coherence Tomography Using Trajectory-Aligned Time-Independent Flow
- **作者**：Veit Hucke, Thomas Pinetz, Gregor Reiter, Ursula Schmidt-Erfurth, Hrvoje Bogunović（维也纳医科大学）
- **发表**：MICCAI 2026（arXiv:2606.18876，v2 于 2026-06-24）
- **核心贡献**：提出 TTA-Flow——基于 flow matching 的生成式测试时自适应框架，通过 (1) 用合成参考轨迹的平均直方图对测试图做**轨迹对齐直方图匹配**、(2) **移除网络的时间条件**，把低成本 OCT 设备（Cirrus/Topcon）的含噪图像重建为接近高 SNR 设备（Spectralis）的图像，供下游分割模型使用；在 IRF/SRF/PED 与 GA 两类 AMD 生物标志物分割上报告 SOTA，且不使用 data-fidelity 项。

## 2. 复杂度评估

- **综合难度**：⭐⭐⭐（困难）——代码可得但无权重、无下游模型、无基线；数据需授权；端到端训练成本高。
- **代码可用性**：低—中。官方仓库开源（MIT，41 文件，约 1600 行 Python），**生成模型侧（训练 + 推理 + 直方图匹配 + ODE 求解）完整可运行**；但**不含预训练权重、不含下游分割模型、不含任何对比基线**。
- **计算资源需求**：高。论文报告 flow matching 模型训练约 **80 小时 / 单张 RTX A6000**（30 万次迭代）；下游分割模型与基线方法需额外训练；推理约 2.2 s/B-scan（128 B-scan/volume ⇒ 约 4.7 min/volume）。
- **数据获取难度**：中—高。RETOUCH 需在 grand-challenge 注册账号、签署数据共享协议后下载（实测 `/Download/` 返回 **HTTP 403**）；自建 GA 配对数据集为维也纳医科大学内部数据，**无公开获取途径**。
- **方法复杂度**：中。U-Net 骨干 + 标准 flow matching 回归损失，属常见训练流程；自定义组件为直方图匹配与参考轨迹统计（仓库已实现）。
- **论文细节完整度**：中。主要超参齐全，但**论文正文与仓库默认值存在多处不一致**（见 §9 风险 R1），且下游分割模型的训练细节（优化器、轮数、损失权重）论文未给出。

## 3. 复现目标（按优先级排序）

### P0（核心，必须完成）

- [ ] **P0-1｜生成模型训练**：在 RETOUCH Spectralis 子集（24 volumes）上训练无条件 flow matching 模型（`remove_noise_conditioning=True`）。— 复现的一切前提，仓库 `src/train.py + train.sh` 直接支持。
- [ ] **P0-2｜参考轨迹与直方图标定**：用训练好的模型采样 n 条参考轨迹，统计各时间步平均直方图。— 仓库 `test.py::generate_reference_trajectories` 已实现，属一次性开销（论文称约 231 s）。
- [ ] **P0-3｜Cirrus → Spectralis 重建 + 下游 DSC**：复现 **Table 1 的 Cirrus 数据块**（IRF / SRF / PED / Mean DSC），目标值 `58.8 / 66.3 / 50.6 / 58.6`（Ours_uncond.）。— 论文主结论所在，验证方法有效性的关键。
- [ ] **P0-4｜Topcon → Spectralis 重建 + 下游 DSC**：复现 **Table 1 的 Topcon 数据块**（除 GA、FID 列），目标值 `51.6 / 52.2 / 49.2 / 51.0`。— 跨设备泛化的第二个证据点。

> 注：P0-3/P0-4 需要**自行实现并训练下游分割模型**（UNet++ + ResNet-18 编码器），仓库未提供；这是 P0 中最主要的额外工作量。

### P1（重要，建议完成）

- [ ] **P1-1｜消融：直方图匹配模块**：令 `test_parameters.num_reference_trajectories=0`（仓库中该值即为关闭直方图匹配的开关），对照普通 flow matching（论文 FM [14] 行）。— 论文称此模块贡献 Cirrus 上 Mean DSC `50.8 → 58.6`，是方法的卖点之一。
- [ ] **P1-2｜消融：时间条件化**：训练 `remove_noise_conditioning=False` 的条件版本，复现 Ours_cond. 行（Cirrus `56.0`、Topcon `49.8`）。— 验证"去时间条件更优"这一反直觉结论。
- [ ] **P1-3｜超参敏感性：s_target**：复现 **Figure 3**，扫描多个参考时间点（论文图中约 0.73–0.88），观察无条件版本的低敏感性（论文称 oracle 选择仅带来约 1 个百分点提升）。

### P2（补充，时间允许时完成）

- [ ] **P2-1｜FID 评估**：用仓库 `util/metrics.py::calculate_fid` + RETFound 特征提取器（权重经 `hf_hub_download` 拉取）计算重建质量 FID。**限制：论文的 FID 是在自建 GA 配对数据上算的（Baseline 550.5 / DDA 148.4 / Ours_uncond. 32.2），该数据不可得**；可行替代是在 RETOUCH 的 Cirrus→Spectralis 上计算同构 FID 作为趋势性证据（需在报告中标注为"非原设定"）。
- [ ] **P2-2｜定性对比图**：复现 **Figure 2** 风格的降噪/病灶保真可视化（输入 vs 重建 vs GT 掩膜）。
- [ ] **P2-3｜监督上界（Upper）**：为 Cirrus/Topcon 训练设备特异的有监督分割模型作为性能上界（论文 Cirrus 61.6 / Topcon 56.8）。— 用于判断复现差距是模型问题还是上界本身差异。

### P3（可选 / 当前判定为不可复现）

- [ ] **P3-1｜GA（geographic atrophy）任务**：**判定不可复现**——数据集为维也纳医科大学内部配对数据（40 患者），论文与仓库均未提供下载途径。替代：寻找公开 GA 数据集（如公开的 GA 分割数据集）做**跨数据集迁移验证**，需用户确认是否接受该偏离。
- [ ] **P3-2｜基线方法 TENT / CoTTA / CPDM / DDA**：仓库未提供，需另行获取或自行实现（CPDM/DDA 各需训练一个扩散模型）。**成本高、收益中**，建议仅在 P0/P1 完成后按需推进。

## 4. 资源情况

### 开源代码

- **仓库地址**：https://github.com/Veit21/tta-flow ✅ **已验证可克隆**（`git ls-remote` 成功，HEAD = `c2f11904e5c571c737e68ef209de09b3b93d0a04`）
- **语言/框架**：Python 3.12 / PyTorch 2.10（`+cu126`）/ Hydra 配置管理 / torchcfm / torchdiffeq
- **规模与许可**：41 个文件、约 1607 行 Python、MIT License、仓库大小约 5.3 MB
- **可用性评估**：**训练与推理链路可直接运行**（`src/train.py`、`src/test.py`、`preferences/` 下 Hydra 配置齐备，含 Dockerfile）；**但仓库内不含任何训练产物（`runs/*` 已被 .gitignore 排除）**
- **已实现的关键组件**：
  - `src/util/flow_matching_util.py::match_cumulative_cdf_batch` —— 直方图匹配（基于 skimage `match_histograms` 改写，支持对 N 张模板图取平均直方图）
  - `src/util/flow_matching_util.py::ODESolver` —— Euler 求解器
  - `src/models/unet.py` —— U-Net 骨干（1.03 亿参数级）
  - `src/util/metrics.py::calculate_fid` + `get_retfound_activations_from_array` —— FID 计算（用 RETFound ViT 作特征提取器）
  - `data/dataset_configs/*.csv` —— 三个设备的官方数据划分（Spectralis 24 / Cirrus 24 / Topcon 22 volumes，与论文一致）
- **缺失组件**（需自行补齐）：预训练权重、下游分割模型（UNet++/ResNet-18、3D-to-2D CNN）、全部对比基线、`.mhd/.raw → .npy` 的数据转换脚本（仓库仅提供 `create_data_configs.ipynb` 用于扫描已有 `.npy`）

### 数据集

- **名称**：RETOUCH（The Retinal OCT Fluid Detection and Segmentation Benchmark）
- **来源**：https://retouch.grand-challenge.org/ ⚠️ **需授权**：需注册 grand-challenge 账号（建议教育邮箱）、签署 data sharing agreement 后方可下载；实测 `https://retouch.grand-challenge.org/Download/` 返回 **HTTP 403**，`/Data/` 返回 404
- **规模**：训练集 70 volumes（Spectralis 24 / Cirrus 24 / Topcon 22），标注 IRF、SRF、PED 三类病灶；B-scan 数：Spectralis 49/vol，Cirrus 128/vol，Topcon 128/vol
- **原始格式**：`.mhd` + `.raw`（SimpleITK 可读），需转换为 `.npy`
- **需要实现的预处理**：
  1. `.mhd/.raw` → NumPy 3D array（`sitk.GetArrayFromImage`，注意 Sitk 的 (z,y,x) 轴序与仓库期望的 (D,H,W) 对齐）
  2. 标签体素值映射到 IRF/SRF/PED 三通道（需核对 RETOUCH 官方 label 编码）
  3. 逐 B-scan 缩放至 **224×224**（论文；仓库默认配置写的是 128×128，见 §9 R1）
  4. 归一化到 **[-1, 1]**
  5. 按设备分目录存放：`volumes/` 与 `annotations/`
- **镜像情况**：已检索 HuggingFace Datasets API（`search=retouch` 返回空）、Zenodo（无匹配）、Kaggle（无公开镜像），**未发现可直接下载的合法镜像**，故数据必须由用户自行取得

### 预训练模型

- **来源**：❌ **无**。已核查：GitHub Releases API 返回空数组；HuggingFace `search=tta-flow` 与 `author=Veit21` 均无结果；仓库存在一条尚未回复的 issue（2026-06-18，0 评论）"Release TTA-Flow checkpoints on Hugging Face"，可确认作者尚未发布权重
- **影响**：**flow matching 模型必须从零训练**，这是本复现计划中最大的单项成本（论文：约 80 h / A6000）
- **例外**：FID 用的 RETFound 特征提取器权重可从 HuggingFace 获取（`retfound_model.py` 内使用 `hf_hub_download`），仅 P2 需要

### 已有复现参考

- **来源**：❌ **未发现任何第三方代码复现**。检索到的相关页面均为论文摘要型内容（themoonlight.io 文献摘要页、alphaxiv 中文摘要、dblp 条目），无实现、无复现博客、无 Papers-with-Code 条目
- **结论**：本计划无"前人踩坑记录"可借，资源核查与配置差异核对需自行完成（§9 已列出已发现的差异）

### 本地运行环境（实测）

| 项目 | 实测结果 |
|---|---|
| Python | 3.12.15 ✅（与仓库要求一致） |
| GPU | ❌ `nvidia-smi: command not found`，本地无 GPU |
| 磁盘可用 | ⚠️ 根分区仅剩约 12 GB |

> **结论**：本地环境不足以执行训练与推理，**必须使用平台的 GPU 会话执行**（数据与代码置于 GPU 侧持久工作目录）。

## 5. 环境配置

- **硬件需求**：
  - **推荐**：单卡 ≥ 24 GB 显存的 NVIDIA GPU（论文用 RTX A6000 48 GB；224×224 输入 + batch 16 + 1.03 亿参数 U-Net 的显存占用需实测，若 OOM 可降 batch 并启用梯度累积）
  - 存储：RETOUCH 转换后约数 GB；训练 checkpoint 与推理输出另需数 GB
- **软件环境**：
  - Python 3.12、PyTorch 2.10.0+cu126、CUDA 12.6+
  - 主要依赖：`torchcfm==1.0.7`、`torchdiffeq==0.2.5`、`hydra-core==1.3.2`、`albumentationsx==2.0.20`、`scikit-image==0.26.0`、`pytorch-lightning==2.6.1`、`timm`（FID 用）
  - 依赖清单已由仓库 `requirements.txt` 固定版本，建议直接使用
- **安装步骤**：
  ```bash
  # 方式 A：直接使用仓库固定依赖（推荐，版本已被作者锁定）
  git clone https://github.com/Veit21/tta-flow && cd tta-flow
  python3.12 -m venv venv && source venv/bin/activate
  pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt

  # 方式 B：Docker（仓库提供 Dockerfile，基于 nvidia/cuda:12.6.3）
  docker build -t tta-flow:latest .

  # 下游分割模型依赖（论文未给，实施时补齐）
  pip install segmentation-models-pytorch monai
  ```
- **额外需要准备的组件**（仓库未含）：
  - RETOUCH `.mhd/.raw → .npy` 转换脚本（自写，属阶段 1）
  - UNet++ + ResNet-18 骨干的下游分割实现（`segmentation-models-pytorch` 或自写）

## 6. 实验步骤

### 阶段 1：环境准备与数据获取

- [ ] 在 GPU 会话中创建项目目录（持久目录），克隆官方仓库并固定到 commit `c2f1190`
- [ ] 安装依赖并验证：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] **（阻塞项）由用户取得 RETOUCH 数据**：注册 grand-challenge → 签署数据共享协议 → 下载 → 上传至 GPU 侧目录
- [ ] 编写并运行 `.mhd/.raw → .npy` 转换脚本；生成 `volumes/`、`annotations/` 结构
- [ ] 校验：Spectralis 24 / Cirrus 24 / Topcon 22 volumes；与仓库 `data/dataset_configs/*.csv` 的条目数逐一对齐
- [ ] 抽查若干 B-scan 与掩膜可视化，确认标签编码正确（IRF/SRF/PED 三类）

### 阶段 2：核心实现

- [ ] **Pilot 冒烟测试**：把训练迭代数压到约 500–2000 步（`train_parameters.max_iterations=2000`）跑通全流程，确认 loss 下降、显存可控、checkpoint 与 TensorBoard 正常
- [ ] 实现参考轨迹与直方图统计的**校验脚本**（对照 `test.py::generate_reference_trajectories`，输出 `[N, T, H, W]` 形状与各步直方图曲线）
- [ ] **实现下游分割模型**：UNet++（ResNet-18 编码器），仅在 RETOUCH Spectralis 子集上训练（论文设定）
- [ ] 实现评测脚本：对重建体积做逐 B-scan 预测，与 GT 掩膜计算 IRF/SRF/PED 的 Dice（注意：只统计被标注的 B-scan）
- [ ] 单元测试：直方图匹配函数的输入输出形状/值域、Dice 计算在已知样例上的正确性

### 阶段 3：实验运行

- [ ] **P0-1**：训练无条件 flow matching 模型（`remove_noise_conditioning=True`，Spectralis，完整迭代数）
- [ ] **P0-2**：生成参考轨迹与平均直方图（记录耗时）
- [ ] **P0-3**：Cirrus → Spectralis 推理 + 下游 DSC 评估
- [ ] **P0-4**：Topcon → Spectralis 推理 + 下游 DSC 评估
- [ ] **P1-1**：关闭直方图匹配（`num_reference_trajectories=0`）重跑推理，得到 FM 对照
- [ ] **P1-2**：训练条件版本（`remove_noise_conditioning=False`）并重跑推理
- [ ] **P1-3**：`s_target` 扫描（复现 Figure 3）
- [ ] **P2**（可选）：FID、定性图、监督上界模型

### 阶段 4：结果分析

- [ ] 汇总为与论文 Table 1 同构的对比表（复现值 vs 论文报告值，含差值）
- [ ] 绘制重建质量对比图与 `s_target` 敏感性曲线
- [ ] 记录与论文不一致之处并归因（配置差异 / 迭代数不足 / 数据预处理差异）
- [ ] 产出复现报告 `results/analysis.md`

## 7. 评估方案

### 评估指标

按论文设定使用 Dice similarity coefficient（DSC，%），仅在被人工标注的 B-scan 上统计；FID 仅用于重建感知质量。

| 指标 | 论文报告值（Ours_uncond.） | 复现目标 |
|---|---|---|
| Cirrus→Spectralis IRF DSC | 58.8 ± 18.6 | 55–62 |
| Cirrus→Spectralis SRF DSC | 66.3 ± 17.2 | 62–70 |
| Cirrus→Spectralis PED DSC | 50.6 ± 27.0 | 45–55 |
| Cirrus→Spectralis Mean DSC | **58.6** | **55–62** |
| Topcon→Spectralis IRF DSC | 51.6 ± 23.1 | 47–56 |
| Topcon→Spectralis SRF DSC | 52.2 ± 16.5 | 47–57 |
| Topcon→Spectralis PED DSC | 49.2 ± 20.9 | 44–54 |
| Topcon→Spectralis Mean DSC | **51.0** | **47–55** |
| 参照：Cirrus Upper（监督上界） | 61.6 | 58–65 |
| 消融：FM 无直方图匹配（Cirrus Mean） | 50.8 | 47–54 |
| 消融：直方图匹配增益（Cirrus Mean） | +7.8（50.8→58.6） | 同向且量级相近 |
| 消融：FID（uncond. vs cond.） | 32.2 / 85.4（自建 GA 数据，**不可得**） | 仅作可选替代实验 |

> **说明**：目标区间是在"无法完全对齐作者硬件与迭代预算"前提下的合理预期，不是精确等值要求。TTA-Flow 的核心可验证论断是**相对关系**——(a) 本方法优于普通 FM；(b) 无条件优于有条件；(c) 对 `s_target` 不敏感——这三条比绝对 DSC 数值更稳健地反映复现成败。

### 基线方法

- **FM [14]**（Li & He, "Back to basics: let denoising generative models denoise"）：仓库内**可通过 `num_reference_trajectories=0` 直接构造**，无需额外实现 ✅
- **Ours_cond.**：仓库内可通过 `remove_noise_conditioning=False` 构造 ✅
- **Baseline（无适配）**：直接用 Spectralis 上训练的分割模型推理低成本设备图像，无需额外模型 ✅
- **Upper（监督上界）**：需为 Cirrus/Topcon 各训练一个监督分割模型 ⚠️ 额外工作量
- **TENT [24] / CoTTA [25]**：需引入第三方实现（均有公开仓库）⚠️
- **CPDM [12] / DDA [5]**：需自行复现或查找作者代码，且各需训练扩散模型 ❌ **成本最高，默认不做**

### 统计分析

- 论文使用 **Wilcoxon 检验（p < 0.05）** 判断 Ours_uncond. 与其最接近竞争者之间的显著性差异，逐患者/逐体积配对比较
- 报告均值 ± 标准差（论文表格中标准差较大，PED 可达 ±27，说明患者间差异显著，需保留该统计口径）
- 训练随机种子固定（仓库默认 `seed: 42`）；如需评估抖动，可在关键配置上重复 2–3 次并报告波动

## 8. 预期输出

```
experiments/
├── code/                      # 实验代码（含官方仓库副本 + 自写的转换/评测/下游脚本）
│   ├── tta-flow/              # 官方仓库（固定 commit c2f1190）
│   ├── preprocess/            # .mhd→.npy 转换、配置生成
│   ├── downstream/            # UNet++ 分割模型训练与评测
│   └── eval/                  # DSC、FID、统计检验
├── results/
│   ├── figures/               # 重建对比图、s_target 敏感性曲线、训练曲线
│   ├── tables/                # 与 Table 1 同构的复现对照表
│   ├── raw_data/              # 训练日志、逐体积 DSC、重建 .npy
│   └── analysis.md            # 复现报告
├── todo.md                    # 实验状态追踪
└── plan.md                    # 本计划
```

## 9. 风险与备选方案

- **R1｜论文正文与官方仓库存在多处配置不一致**（已实测发现，必须在复现中显式记录与取舍）：

  | 参数 | 论文 | 仓库默认值 | 处理建议 |
  |---|---|---|---|
  | EMA decay | 0.999 | `train_parameters.ema_decay: 0.9999` | 以仓库为准（论文正文可能笔误），并在报告中注明 |
  | 积分步数 S | 100 | 训练 101 / 推理 `test_parameters.num_integration_steps: 51` | 两组都跑，报告差异 |
  | 参考轨迹数 n | 100 | `num_reference_trajectories: 50` | 以论文 100 为准，成本允许时对照 50 |
  | 输入分辨率 | 224×224 | `model.params.dim: [1, 128, 128]`（注释写 `[1, 224, 224]`） | **高优先级核对项**：两种分辨率都跑，可能显著影响 DSC |
  | 训练迭代 | 300,000 | `max_iterations: 300001` | 一致，无冲突 |
  | ODE 积分起点 | 从 `s_target` 起步、只跑剩余步数 | `test.py` 中 `s_target` **仅用于选择匹配的目标直方图**，积分仍从 t=0 跑满全程 | **代码与论文描述不一致**，需在复现时验证哪种与论文报告结果对应 |

  → **备选**：若两种设定结果差异显著，以"能复现论文数值的设定"为准，并在报告中明确记为该论文的表述/实现不一致。

- **R2｜数据不可得（阻塞级）**：RETOUCH 需用户注册授权（实测 403）。→ **备选**：(a) 用户自行注册下载后上传；(b) 用户若已有本地副本直接提供；(c) 若两条路都不可行，则本计划的 P0 无法执行，需改为"仅代码级复现"（跑通训练/推理链路 + 在合成或替代 OCT 数据上验证方法机制，明确标注非原设定）——**该决策只能由用户做出，不得自行用替代数据顶替**。

- **R3｜无预训练权重，训练成本高**：30 万次迭代约 80 h/A6000。→ **备选**：先跑 **quick 档**（约 2 万–5 万次迭代）验证完整链路与趋势，再决定是否投入 full 档；若 GPU 时长受限，在报告中把结论限定为"趋势复现"。

- **R4｜自建 GA 数据不可得** → GA 列与 FID 列**判定不可复现**；备选为公开 GA 数据集上的迁移验证（需用户确认口径偏离）或直接裁剪该部分并在报告中说明。

- **R5｜下游分割模型无官方实现**：分割模型的训练细节论文未给出，DSC 绝对值会受此影响。→ **备选**：使用标准 UNet++/ResNet-18 配置（`segmentation-models-pytorch`）并在 Spectralis 内部做交叉验证校准，使 Baseline 与 Upper 均达到合理水平后再比较相对关系。

- **R6｜显存/时间超预算**：224×224 + batch 16 + 1.03 亿参数可能 OOM 或超时。→ **备选**：降 batch + 梯度累积、开混合精度、或降到 128×128 分辨率（同时记录为配置偏离）。

- **R7｜基线方法工作量大**：CPDM/DDA 需各自训练扩散模型。→ **备选**：P0 完成后按边际价值决定是否复现；至少保留 FM / Ours_cond. / Baseline / Upper 四个"仓库内可得"的对照。

## 10. 工作量估计

| 阶段 | 复杂度 | 说明 |
|---|---|---|
| 环境准备 | 轻量 | 依赖已由 `requirements.txt` 锁定，Dockerfile 齐备；主要耗时在仓库克隆与镜像拉取 |
| 数据获取与预处理 | **重度** | 需用户完成注册授权；`.mhd/.raw → .npy` 转换 + 轴序/标签编码核对需谨慎，易出错 |
| 核心实现 | 中等 | 生成侧开箱可用；**下游 UNet++ 分割模型与评测脚本需从零实现**（约数百行） |
| 实验运行 | **重度** | full 档 flow matching 训练约 80 h/A6000；推理约 2.2 s/B-scan（Topcon 22 volumes × 128 B-scan ≈ 1.6 h）；下游模型训练与消融另计 |
| 结果分析 | 中等 | 需产出与 Table 1 同构的对照表、敏感性曲线，并对 R1 中的配置差异做归因分析 |

**建议执行节奏**：
1. **先做 quick 档端到端闭环**（小迭代数 + 单设备），把数据管线、下游模型、评测脚本全部跑通 —— 这一步能暴露 80% 的工程风险；
2. 再决定是否投入 **full 档** 30 万次迭代；
3. 消融与基线按边际价值排序推进。

---

**执行前提（需用户确认）**：
1. RETOUCH 数据的获取方式（自行注册下载后上传 / 已有本地副本 / 暂不获取）；
2. 是否接受先跑 quick 档验证链路，再决定 full 档投入；
3. GPU 会话的可用时长预算（决定 full 档训练能否完成）。