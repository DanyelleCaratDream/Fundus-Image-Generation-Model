# 范文调研与论文定位

> 建立于 2026-10-15。基于三路网络检索（医学影像生成泛 / 眼底+眼科 / 评估方法论）共 30+ 篇文献。
> **目的**：在动笔前搞清楚——我们的候选贡献哪些已经被人做过、哪些还有空间。
> ⚠️ 检索非穷尽。凡标「待核实」者未逐字核到原始出处，引用前请再核对。

---

## 一、坏消息：三个候选贡献点，都已有前人

### ❌ 1.「通用指标看不出医学语义差异」——已是学界共识，不是新发现

**至少 5 篇已明确说过：**

| 论文 | 结论 | 出处 |
|:--|:--|:--|
| **Deo et al. 2025**《Metrics that Matter》 | 用受控实验证明 FID/KID/MMD **对局部临床相关形态改变不敏感**，且**上游指标对模型的排名与下游任务不一致** | arXiv:2505.07175 |
| **Jayasumana et al. 2024**《Rethinking FID》(CVPR) | FID 三大硬伤：Inception 特征对现代生成内容表征差、**2048 维特征的高斯假设错误**、小样本方差极大；实证与人类评分相矛盾 | arXiv:2401.09603 |
| **Jung & Keuper 2021** (NeurIPS Workshop) | FID 对**亮度/饱和度/对比度扰动不敏感**——因为 Inception 特征内置了自然图像偏差。而医学病灶恰恰是这类细微局部变化 | arXiv:2110.11875 |
| **Chong & Forsyth 2020** (CVPR) | FID 与 IS 都是**有偏估计**，样本越少偏差越大 | arXiv:1911.07023 |
| **TVST 视网膜域研究** ⚠️ | 两位视网膜专科医生用 ETDRS 四级量表盲评合成图，**专家分级与 PSNR/SSIM 的排序不一致** —— 这在**视网膜域**已经验证过了 | tvst.arvojournals.org（待核实卷期） |

> **Deo 2025 是最直接命中我们发现的论文。** 我们的"FID 相近但出血保留率 0.254 vs 1.137"是它结论的一个新实例。
> **把它当核心贡献会被拒。**

### ❌ 2.「血管 mask 条件生成眼底图」——2017 年就有人做，而且数据比我们少

| 论文 | 做法 | 数据量 |
|:--|:--|:--|
| **Guibas 2017** | 两阶段：先造血管 mask → 再生成彩图 | **DRIVE 仅 40 对** |
| **Costa et al. 2018** (IEEE TMI) | 先学血管树分布 → 条件 GAN 由血管图生成彩图 | 训练 **614 对** |
| **DR-GAN 2022** (JBHI) | 血管 + 视盘 + 病灶 mask + 分级向量作条件 | EyePACS 3.5万 + FGADR 1842 |
| **RetinaGAN 2023** (BOE) | 病灶 mask 条件 + SPADE 生成，**代码公开** | 清洗后 **1400 张** |

> **Costa 2018 用 614 对训练数据发上了 IEEE TMI** —— 说明小数据不是死路，关键在怎么做和怎么评。

### ✅ 3.「病灶级保真度指标」——**已核实：不撞车**（2026-10-15 读完全文后更正）

原本标为最高风险，**读完 Asokan & Manju 2026 全文后确认：不是同一个东西。**

**LFS 的准确做法**（原文附录 A）：
- 架构是 **judge–artist–student**：多病灶 U-Net 分割器（judge）→ Pix2Pix 条件生成（artist）→ ResNet-50 分类（student）
- 流程：Pix2Pix 由 4 通道病灶 mask 生成图 → **U-Net 对生成图重新分割** → 与**输入的条件 mask** 逐通道算 **Dice**，再对出现的通道求平均

**与我们出血保留率的三条本质区别：**

| | 我们 | LFS |
|:--|:--|:--|
| **参照物** | **真实病灶**（真实图 vs 生成图） | **条件 mask**（生成图重分割 vs 输入 mask）|
| **度量算子** | **Wasserstein 距离 + 面积保留比**（分布/几何层面）——能捕捉"整体缩小/溶解/漂移" | **Dice 重合度**（像素重叠）——**对"整体偏小偏淡但形状尚可"不敏感** |
| **病灶范围** | **仅出血（HE）** | **MA/HE/EX/SE 四通道平均** |

**次级差异**：LFS 定位是**样本筛选器**（排序扔掉低分图）；我们是**评估标尺**（诊断生成器有没有病灶溶解）。LFS 是**逐样本分数**；我们是**每模型一个聚合标量**。

> **可诚实区分定位。但两条纪律：**
> 1. **不要说"我们和 LFS 一样"，也不要说"我们首创病灶级指标"**
> 2. ⚠️ **Asokan 全文从未提 FID**（grep 过，"FID" 只是 "lesion FIDelity" 的子串）。他们批评的是"global realism metrics"泛称。**引用时必须按原文口径，不能写成"他们批评 FID"**

**⚠️ 另需知道：Asokan 自己的结果很弱**，可作为我们的参照系：
- held-out test **仅 N=17**；judge 的分割 Dice 只有 **0.0733**（作者自己承认很低）
- 主指标 QWK 均值 **0.065 ± 0.090**，纯真实 baseline 是**负的**（−0.122）
- 作者自称 **"controlled proof-of-concept under limited data conditions"**
- 他们发现 **"Low-LFS paradox"**：低保真度样本反而拿到更高的 Macro F1（当作 hard 样本提升少数类召回）→ 保真度与鲁棒性存在 trade-off

> 这条 paradox 值得记一笔——如果我们的实验也观察到类似现象，可以与之对话。

**另一个竞争者**：**Konz et al. 2026**（*Medical Image Analysis*）的 **FRD**——用放射组学特征替代 Inception 特征算 Fréchet 距离，是**分布级**指标。我们是**病灶实例级**，可区分。

---

## 二、好消息：一个真正的空白

### ✅「多方法统一口径横评」——调研范围内没人做过

看这 30+ 篇的组织方式，**全部是"提出一个方法 → 证明它好"**：

- DR-GAN：提出新生成器 → 比过 baseline
- RetinaGAN：提出两阶段 → 比过 FID
- Pandey 2025：提出增广方案 → 比过不加
- Konz FRD：提出新指标 → 比过 FID

**没有一篇是"把 VAE / GAN / 扩散 / 传统 ML 放在同一套协议下系统横评，诚实报告谁行谁不行、为什么不行"。**

而我们手上的恰恰是这个：

- **6 个深度方法**（VAE / DCGAN / 4 个 DDPM 变体）
- **5 个传统 ML 方法**（PCA / GMM / 补丁 / 泊松 / Retinex）
- **同一套两层指标**（通用 11 项 + 眼底专用自设计）+ **六维加权评分**
- **一个受控消融**：同架构只换损失函数 → 出血保留率 0.254 vs 1.137

**这可能是论文真正的落脚点。**

---

## 三、可照抄的模板（这是本次调研最实用的产出）

### 3.1 贡献列表怎么写（原文抄录）

**Frid-Adar et al. 2018**（Neurocomputing，TSTR 经典）：
> 1) Synthesis of high quality focal liver lesions from CT images using GANs.
> 2) Design of a CNN-based solution for the liver lesion classification task, with comparable results to state-of-the-art methods.
> 3) Augmentation of the CNN training set, using the generated synthetic data - for improved classification results.

**Khader et al. 2023**（Sci Rep，5 条）：
> (i) 提出用 VQ-GAN 潜空间生成高分辨率 3D 图像的方法…(ii) 在四个公开数据集上验证，数据量有限（n ≤ 1844），**代码公开**…(iii) 优于 reference GAN，多样性更好、不易模式崩溃…(iv) **已确立的指标可能无法像人一样评估图像**，故请放射科医生评估…(v) 证明合成数据在低数据场景下的潜力…

**Konz et al. 2024**（FRD，3 条）：
> 1. We highlight the shortcomings of common metrics for medical image distribution comparison (e.g., FID)…
> 2. We introduce Radiomic Distance (RaD)… offers (1) alignment with downstream tasks, (2) stability for small datasets, (3) clinical interpretability.
> 3. We validate RaD across diverse datasets and applications…

**共同规律**：①新方法/新指标 → ②多数据集验证 → ③**下游任务或专家验证** → ④**代码开源**。
**第 ③④ 条对我们最重要。**

### 3.2 Introduction 三段式

> ① 临床需求（为什么这问题重要）→ ② 现有方法缺陷（**必须带引用**）→ ③ 本文贡献列表

### 3.3 医生评分表设计（可照抄）

| 来源 | 设计 |
|:--|:--|
| **DR-GAN 2022** | 3 位眼科医生独立打分：**① fidelity 1–10 分**（它 7.97，真人真图 8.89，baseline 5.36）；**② 真/假判别准确率**（它 65.8%，越接近 50% 越像）；**③ 让医生给合成图打 DR 分级，算分级准确率**（85.3%） |
| **Pandey 2025** (BJO) | **8 位专科医生**，盲评 96 张（48 合成 + 48 真实），只报**判别真假准确率**（61.1%） |
| **Kim 2022** (Sci Rep) | Turing test + **解剖学定量**：血管量差 0.43%、平均 SNR 差 1.5% |
| **Liu et al. 2023** (PMB) | 形式化的"理想观察者"框架 + 放射科医生盲评 |

### 3.4 TSTR 标准做法

**原始出处：Esteban et al. 2017**（arXiv:1706.02633，首次提出 TSTR/TRTS）

标准流程：
1. 生成带标签的合成训练集
2. 只用合成集训练下游模型
3. 在**真实**留出测试集上评估
4. 同时跑 **TRTR**（真实训、真实测）作为**上界基线**
5. 报下游指标（分类报 AUC/Acc，分割报 Dice），**以"相对 TRTR 基线的性能下降幅度"为核心结论**
6. 近年医学综述强调：应做**多次随机切分 + 报告均值±标准差**

**医学眼底域的实践范例——Pandey et al. 2025**：
- 用生成图**只补最缺的那个类**（AMD 补到 238 张，与真实样本等量）
- 真实测试集上 **F-score +5.3%、accuracy +5.8%（p<0.01）**
- **补多了反而过拟合**

### 3.5 评估必须三层（缺一层被质疑）

```
① 像素/分布指标（FID/KID/IS/P-R …）
② 下游任务（分类/分割，TSTR）
③ 专家盲评（reader study）
```

### 3.6 小数据集论文的通用套路

> ① 坦诚说明数据少 → ② 用生成做数据增广 → ③ **只补最缺的类** → ④ 用下游指标（而非仅 FID）证明有效

**小数据成功的四种打法**（从这些论文里提炼）：
1. 降分辨率（256×256）+ 两段式（先出 mask 再出图）
2. 用前置分割模型把图变成条件图，把"生成"降级为更容易的 image-to-image
3. **用更大的无关数据池增加多样性，小数据只训条件端**
4. 大量依赖预训练/迁移（Kim 用 27 万正常图预训练 → 迁移到只有 2671 张的罕见病）

---

## 四、定位建议

### 4.1 不能立的说法

- ❌「我们发现了 FID 在医学影像上不可靠」——已被发表多次
- ❌「我们用血管 mask 条件生成了眼底图」——2017 年就有
- ❌「我们提出了病灶保真度指标」——Asokan & Manju 2026 已做（除非能明确划界）

### 4.2 可以立的说法

**方向一：系统横评（最有把握）**
> 首个把 **6 个深度生成方法 + 5 个传统 ML 方法**放在**同一套两层评估协议 + 六维门控评分**下进行的小样本眼底生成系统对比，并诚实报告失败机理。

**方向二：病灶级指标 + 因果验证（需补实验）**
> 不是"发现 FID 不行"，而是**提出一个病灶语义级指标，并系统证明它能预测下游 DR 分级的临床效用，而 FID 不能**。

这正是 Deo et al. 2025 指出的空白——他们只做了**敏感性分析**，**没有提出可用的病灶级指标**。

**与两个竞争者的划界**：
| | 它们是什么 | 我们怎么区分 |
|:--|:--|:--|
| LFS (Asokan 2026) | 基于分割一致性的**样本筛选分数** | 我们若定位为**生成器评估**、面向**多类病灶**、且**与临床分级直接挂钩**，则用途不同 |
| FRD (Konz 2026) | **分布级**放射组学距离 | 我们是**病灶实例级** |

### 4.3 最关键的判断

> **Phase D（下游验证）不是可选项，是入场券。**
>
> 本次调研的 30+ 篇里，凡是声称"生成的图有用"的，**全都做了下游任务验证 + 专家盲评**。没有一篇只靠 FID 就发出去的。
>
> 我们的 Phase D 目前是**零**。这必须补。

---

## 五、对标论文宽目录（当参考文献库用）

### A. 医学影像生成（泛）
| # | 论文 | 年份/出处 | 为什么值得看 |
|:--:|:--|:--|:--|
| 1 | GAN-based Synthetic Medical Image Augmentation…Liver Lesion | 2018 Neurocomputing | **TSTR 经典范式**，含医生盲评 |
| 2 | GANs for Medical Image Synthesis: An Empirical Study | 2023 J. Imaging | **多模型系统评测**范例；实测"FID 与下游 Dice 不相关" |
| 3 | Diffusion Models in Medical Imaging: A Comprehensive Survey | 2023 MedIA | 扩散综述，taxonomy 结构极佳 |
| 4 | DDPM for 3D medical image generation | 2023 Sci Rep | 小数据集（≤1844）+ 医生 Likert 盲评 |
| 5 | Medfusion: multimodal comparison of LDM vs GAN | 2023 Sci Rep | **含眼底数据**；警示 FID/KID 非为医学设计 |
| 6 | Synthetic Tumors Make AI Segment Tumors Better | 2022 NeurIPS Workshop | 生成直接服务下游分割 |
| 7 | Anatomically-Controllable Medical Image Generation (seg-guided) | 2024 MICCAI | **条件可控生成**，与我们同构 |
| 8 | FRD: Fréchet Radiomic Distance | 2024 arXiv / 2026 MedIA | ⚠️ 竞争者：医学化指标 |
| 9 | Metrics that Matter | 2025 arXiv | ⚠️ **最直接命中我们的发现** |

### B. 眼底 / 眼科
| # | 论文 | 年份/出处 | 为什么值得看 |
|:--:|:--|:--|:--|
| 10 | End-to-End Adversarial Retinal Image Synthesis | 2018 IEEE TMI | **614 对**训练；ISC + 互信息证明新颖性 |
| 11 | Synthetic Medical Images from Dual GANs | 2017 arXiv | 两阶段思想源头；**DRIVE 仅 40 对** |
| 12 | DR-GAN: Conditional GAN for Fine-Grained Lesion Synthesis | 2022 JBHI | **评估体系最完整**，医生评分表可照抄 |
| 13 | Synthesizing realistic high-resolution retina by style-based GAN | 2022 Sci Rep | Turing test + 解剖学定量；**27 万预训练→2671 迁移** |
| 14 | Generation of Structurally Realistic Fundus Images with Diffusion | 2024 arXiv | 扩散 + A/V mask 条件 |
| 15 | RetinaGAN (High-Fidelity DR Fundus from Freestyle Lesion Maps) | 2023 BOE | **代码公开**，最易复现对比 |
| 16 | Realistic fundus photograph generation for improving classification | 2025 BJO | ⭐ **最贴近我们的黄金范式** |
| 17 | RetinaLogos | 2025 arXiv | 140 万 caption 数据，SOTA 天花板 |
| 18 | RLAD: Layout-Aware Generative Modelling | 2025 arXiv | 部分给定条件 + 泛化提升；数据公开 |
| 19 | Controllable retinal synthesis via conditional StyleGAN | 2024 arXiv | 数据+代码公开，复现成本低 |
| 20 | Repurposing the Image Generative Potential: Exploiting GANs to Grade DR | 2024 CVPRW | **小数据巧思路**：复用判别器做分类 |
| 21 | Asokan & Manju: Lesion-Fidelity-Aware Synthetic Data Augmentation | 2026 IEEE Access | ⚠️ **直接撞车，必读** |

### C. 评估方法论
| # | 论文 | 年份/出处 | 为什么值得看 |
|:--:|:--|:--|:--|
| 22 | Rethinking FID (CMMD) | 2024 CVPR | FID 三大硬伤的理论根据 |
| 23 | Effectively Unbiased FID and IS | 2020 CVPR | FID/IS 有偏估计的统计学根据 |
| 24 | Internalized Biases in FID | 2021 NeurIPS Workshop | 解释为何 FID 对病灶不敏感 |
| 25 | Evaluation metrics for synthetic medical imaging（综述） | 2026 EJR | 47 篇综述；"FID 抓不到稀有病灶" |
| 26 | Evaluation metrics in medical imaging AI（手册式综述） | 2025 EJR-AI | 引言可引 |
| 27 | Observer-study-based approaches（reader study 方法） | 2023 PMB | 评分表设计参考 |
| 28 | UWFA 合成图专家评估（视网膜域） | TVST（待核实） | ⚠️ 视网膜域已发现专家与指标排序不一致 |

### D. TSTR 与记忆检测
| # | 论文 | 年份/出处 | 为什么值得看 |
|:--:|:--|:--|:--|
| 29 | Real-valued Time Series Generation with RCGAN（**TSTR 原始论文**） | 2017 arXiv | 必引 |
| 30 | Extracting Training Data from Diffusion Models | 2023 USENIX Sec | 复制检测的理论动机 |
| 31 | Unconditional LDM memorize patient imaging data | 2025 Nat. Biomed. Eng. | ⭐ **医学域复制检测对标**：LDM 比 GAN/VAE 更易记忆 |
| 32 | A Non-Parametric Test to Detect Data-Copying | 2020 AISTATS | 复制检测的形式化定义 |

---

## 六、立即行动项

| 优先级 | 事项 | 为什么 |
|:--:|:--|:--|
| ✅ **已完成** | ~~读 Asokan & Manju 2026~~（2026-10-15 已读全文） | **结论：不撞车**，差异已写入第三节 |
| 🔴 **P0** | **读 Deo et al. 2025（arXiv:2505.07175）** | 最直接命中我们的发现，必须引用并划界 |
| 🔴 **P0** | **读 Pandey et al. 2025（BJO 109(7):791-798）** | 最贴近的黄金范式，Phase D 照它做 |

### 三篇 P0 论文的获取方式（2026-10-15 核实，**均可免费读全文**）

| 论文 | 链接 | 备注 |
|:--|:--|:--|
| **Asokan & Manju 2026** | 文章页 https://ieeexplore.ieee.org/document/11510187 ｜ **全文 PDF** https://ieeexplore.ieee.org/iel8/6287639/11323511/11510187.pdf ｜ DOI https://doi.org/10.1109/ACCESS.2026.3690867 | IEEE Access 是 **GOLD Open Access（CC BY-NC-ND 4.0）**，全文免费 |
| **Deo et al. 2025** | arXiv https://arxiv.org/abs/2505.07175 ｜ HTML https://arxiv.org/html/2505.07175v1 ｜ **代码** https://github.com/YashDeo-York/GenMed | 完全开放 |
| **Pandey et al. 2025** | BJO https://bjo.bmj.com/content/109/7/791 ｜ PDF https://bjo.bmj.com/content/bjophthalmol/109/7/791.full.pdf ｜ PubMed https://pubmed.ncbi.nlm.nih.gov/39939121/ | CC BY-NC 4.0，全文免费 |

> 下载后放进 `literature/pdf/`，命名建议 `Asokan2026-LesionFidelityScore.pdf` 这样。
| 🟠 P1 | 读 Costa 2018（IEEE TMI）+ Kim 2022（Sci Rep） | 学小数据怎么成功 |
| 🟠 P1 | 把 `literature/pdf/` 建起来，下载上述 P0/P1 论文 | 见 `literature/README.md` |
| 🟡 P2 | 用 Zotero/`.bib` 管理引用，边读边录 | 避免后期返工 |
| 🟡 P2 | 设计眼科医生评分表（照抄 DR-GAN 的三项设计） | 现在定下来，生成图片时同步做 |
