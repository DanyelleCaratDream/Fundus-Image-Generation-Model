# 条件流匹配（CFM）阅读清单

> 2026-10-21 调研。按"奠基 → 推广 → 规模化 → 医学应用"排序。
> 状态：`未读` / `在读` / `已读`。所有链接均已核对可访问。

## A. 奠基三篇（ICLR 2023 同期独立提出，方法族源头）

| # | 论文 | Venue | 链接 | 核心贡献 | 与本项目关系 |
|:--|:--|:--|:--|:--|:--|
| A1 | **Flow Matching for Generative Modeling**<br>Lipman, Chen, Ben-Hamu, Nickel, Le | **ICLR 2023** | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) · [OpenReview](https://openreview.net/forum?id=PqvMRDCJT9t) | 提出 FM/CFM：回归条件向量场、simulation-free 训练 CNF；证明 CFM 与不可解边际 FM 同梯度 | **我们的实现直接对应这篇**（OT 线性路径 + 速度回归） |
| A2 | **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**<br>Liu, Gong, Liu | **ICLR 2023** | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) | 直线插值 + **reflow** 迭代拉直轨迹，实现极少步采样 | 若 F2 采样步数不够，reflow 是下一步优化 |
| A3 | **Building Normalizing Flows with Stochastic Interpolants**<br>Albergo, Vanden-Eijnden | **ICLR 2023** | [arXiv:2209.15571](https://arxiv.org/abs/2209.15571) | 随机插值统一框架，覆盖 flow 与 diffusion | 理论统一视角，写 Related Work 用 |

> 三篇是"同一方法族的三条入口"，ICLR 2025 官方博客明确这样叙述
> （[A Visual Dive into Conditional Flow Matching](https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/)）。

## B. 推广与几何改进

| # | 论文 | Venue | 链接 | 核心贡献 | 与本项目关系 |
|:--|:--|:--|:--|:--|:--|
| B1 | **Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport (OT-CFM)**<br>Tong et al. | **TMLR 2024**（Expert Certification） | [arXiv:2302.00482](https://arxiv.org/abs/2302.00482) · [OpenReview](https://openreview.net/forum?id=CD9Snc73AW) | minibatch OT 耦合替代独立耦合 → 轨迹更直、训练更稳、推理更快 | **F2 若要升级，优先试 OT 耦合**（TorchCFM 现成） |
| B2 | **Multisample Flow Matching: Straightening Flows with Minibatch Couplings**<br>Pooladian et al. | **ICML 2023** | [arXiv:2304.14772](https://arxiv.org/abs/2304.14772) | 与 B1 同期，minibatch coupling 拉直 | 同上，可作对照 |
| B3 | **Flow Matching Guide and Code**<br>Lipman, Havasi, Holderrieth, Shaul et al. | arXiv 2024（Meta 官方指南） | [arXiv:2412.06264](https://arxiv.org/abs/2412.06264) · [文档](https://facebookresearch.github.io/flow_matching/) | 教科书式统一指南 + 官方 PyTorch 库 | **写论文方法节的第一参考**；库已克隆 |

## C. 规模化：现代图像生成主流

| # | 论文 | Venue | 链接 | 核心贡献 | 与本项目关系 |
|:--|:--|:--|:--|:--|:--|
| C1 | **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（Stable Diffusion 3）**<br>Esser et al. | **ICML 2024**（Best Paper） | [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) | rectified flow + MMDiT；大规模证明 RF 优于既有扩散配方；改进噪声采样（logit-normal） | **"RF 是当代主流"的最强证据**，Introduction 必引 |
| C2 | **Taming Rectified Flow for Inversion and Editing** | ICML 2025 | [ICML page](https://icml.cc/virtual/2025/poster/43737) | RF 模型的反演与编辑 | 若做"生成图可控编辑"再读 |

> 工业界延伸（非论文）：FLUX、SD3.5 均基于 rectified flow——说明方法族不是小众。

## D. 医学影像应用（与我们的任务最接近）

| # | 论文 | Venue | 链接 | 核心贡献 | 与本项目关系 |
|:--|:--|:--|:--|:--|:--|
| D1 | **Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality（MOTFM）**<br>Yazdani et al. | **MICCAI 2025** | [arXiv:2503.00266](https://arxiv.org/abs/2503.00266) · [项目页](https://milad1378yz.github.io/MOTFM/) · [代码](https://github.com/milad1378yz/MOTFM) | OT-FM 加速医学图像合成；支持 **unconditional / class-conditional / mask-conditional**，2D/3D；合成数据可提升下游分类/分割 | **范式最贴近我们**：mask 条件 + 下游分类价值，可直接对标写法 |
| D2 | **RetinaLogos: Fine-Grained Synthesis of High-Resolution Retinal Images** | arXiv 2025 | [arXiv:2505.12887](https://arxiv.org/abs/2505.12887) | 视网膜 CFP 细粒度文本到图像合成（140 万图 + caption） | 眼底专用，方法待读确认（是否用 FM）；Related Work 用 |
| D3 | AlphaFold 3（Abramson et al.） | **Nature 2024** | [Nature](https://www.nature.com/articles/s41586-024-07487-w) | 结构预测用 flow matching 做生成头 | 方法族跨领域影响力证据（一句话引用即可） |

## E. 外部代码库（已克隆 / 待下载）

| 库 | 用途 | 地址 |
|:--|:--|:--|
| facebookresearch/flow_matching | Meta 官方库（连续+离散 FM） | https://github.com/facebookresearch/flow_matching |
| atong01/conditional-flow-matching（TorchCFM） | OT-CFM 等变体，`pip install torchcfm` | https://github.com/atong01/conditional-flow-matching |
| milad1378yz/MOTFM | 医学 OT-FM 参考实现（含 mask 条件） | https://github.com/milad1378yz/MOTFM |
| gnobitab/RectifiedFlow | Rectified Flow 官方实现（reflow） | https://github.com/gnobitab/RectifiedFlow |
| Stability-AI/generative-models | SD3 rectified flow 实现 | https://github.com/Stability-AI/generative-models |
| dongzhuoyao/awesome-flow-matching | 论文索引（持续更新） | https://github.com/dongzhuoyao/awesome-flow-matching |

## G. 2026-10-23 新增精读（作者提供 PDF，详见 notes/01-lit-review-2026-10-23.md）

| # | 论文 | 类型 | 核心方法 | 代码 |
|:--|:--|:--|:--|:--|
| G1 | **TTA-Flow**: Test-Time Adaptation in OCT Using Trajectory-Aligned Time-Independent Flow | MICCAI 类 | FM + 轨迹对齐直方图匹配 + 去时间条件 + 预测 x1 | [tta-flow](https://github.com/Veit21/tta-flow) |
| G2 | **FSFM**: Few-Shot Flow Matching with Stochastic Barycentric Sampling | MICCAI 2026 | 检索近邻 + Dirichlet 重心条件，DR 眼底少样本 | 匿名仓库已失效（410） |
| G3 | **SymmFlow**: Symmetrical Flow Matching | AAAI 2026 | 双向流统一生成/分割/分类 | [SymmetricFlow](https://github.com/caetas/SymmetricFlow) |
| G4 | Self-evolving AI virtual cells（综述） | 综述 | 提及 CellFlow、AlphaFold3 | — |
| G5 | **Tiger-Model**（罕见甲状腺癌） | Nat. Commun. 2025 | 文本引导扩散增广 | [Tiger-Model](https://github.com/fangdai-dear/Tiger-Model) |

## F. 给作者的检索关键词

中文库/英文库都能用，建议组合：

1. `conditional flow matching`（核心）
2. `rectified flow` / `reflow`
3. `stochastic interpolants`
4. `optimal transport flow matching` / `minibatch optimal transport`
5. `flow matching medical image synthesis`
6. `flow matching retinal` / `fundus` / `ophthalmology`
7. `flow matching few-shot` / `limited data` / `small dataset`
8. `flow matching vs diffusion`（对比评测）
9. `flow matching classifier-free guidance`
10. `flow matching image-to-image translation` / `inpainting` / `super-resolution`

**优先 venue**：NeurIPS / ICLR / ICML / CVPR / ICCV / MICCAI / MIDL / TMLR。
**特别想要**：① 小样本/少数据下的 FM 表现（我们的 193 张是核心约束）；
② 眼底/视网膜 FM 生成（确认是否已有人做过，避免撞车）；
③ FM 与 diffusion 在**同一数据同一预算**下的受控对比（我们要写"失效机理链"，需要这类证据）。
