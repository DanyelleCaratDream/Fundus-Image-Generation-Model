# 文献综述②：眼底特异生成 + 新奇算法组合

> 调研目的：专门盘点眼底彩照生成方向的方法 + 可以混搭的新奇算法组合。
> 边界：§2 的"结构/条件驱动生成"是深度范式，这里只作为**结构先行的证据链引用**（非本目录实现内容），供后续借鉴；深度生成与预训练归属见 `00-README.md` 边界表。
> 验证说明：所有引用经在线检索验证存在，无编造；标注"需实验验证"的是头脑风暴，无直接文献。
> 更新日期：2026-08-03

---

## 结论先行（TL;DR）

1. **非深度眼底合成确实存在，且是深度方法出现前的主流**，核心范式恰好就是"血管骨架/结构生成 + 纹理合成"两步走——这正是老师要求的非深度方向。
2. **最有文献直接支撑的两条新奇组合**：**泊松病变重排**（Yu et al. 2021）和 **Retinex 光照交换**（Zhang et al. 2022）。两者都天然产生"相似但不相同"的图像，且几乎不改诊断语义。
3. **330 张属于"记忆/复制高风险区"**：文献给出复制率随数据集规模指数衰减的明确结论。任何方法都应配套"最近邻复制检测三件套"。

---

## 1. 经典/非深度眼底图合成方法（有文献支撑）

### 1.1 补丁拼布（Image Quilting）合成眼底背景 + 血管纹理
- **原理**：从真实眼底图无血管区采集数十万小贴片（7×7），用 Correspondence Map 约束、按 Efros-Freeman 拼布（最小误差边界切割）逐块拼出背景；同法沿血管中心线拼血管纹理。
- **文献**：
  - Fiorini, De Biasi, Ballerini, Trucco, Ruggeri（2014）"Automatic Generation of Synthetic Retinal Fundus Images"，STAG 2014，**DOI 10.2312/stag.20141238.041-044**（[EG Digital Library](https://dlold.eg.org:443/handle/10.2312/stag.20141238.041-044)）[V]
  - 方法学根基：Efros & Freeman 2001（Image Quilting）、Efros & Leung 1999（非参数纹理合成）
- **330 张 KW IV 可行性**：高。330 张足够建贴片词典；但 KW IV 渗出/出血遮挡多，"无血管净区"贴片要谨慎筛选（绿色通道 + 病变掩膜外采样）。

### 1.2 模型化 + 统计血管树合成（ASM 形状 + Kalman 滤波纹理）
- **原理**：Active Shape Model 对真实血管中心线做 PCA（保留 98% 方差），从多元正态采样形状参数生成新骨架；分叉位置按真实分叉密度图采样，管径服从 Murray 分叉法则；血管灰度剖面用 6 参数 Hermite 模型 + Kalman 滤波逐剖面生成，保证与前后邻居及背景统计一致。
- **文献**：
  - Bonaldi, Menti, Ballerini, Ruggeri, Trucco（2016）"Automatic Generation of Synthetic Retinal Fundus Images: Vascular Network"，SASHIMI/MICCAI 2016，LNCS 9968，**DOI 10.1007/978-3-319-46630-9_17**（[Springer](https://rd.springer.com/chapter/10.1007/978-3-319-46630-9_17)）[V]
- **可行性**：中高。330 张可建 ASM，但 KW IV 渗出会污染中心线提取——先用现成分割器提取干净骨架再建模。

### 1.3 过程化/生物物理血管生成（DLA / L-system / CCO）
- **原理**：用分形/优化过程直接长出一棵血管树。
- **文献**：
  - Family, Masters, Platt（1989）"Fractal pattern formation in human retinal vessels"，*Physica D* 38:98-103——实测人眼血管分形维数 ≈1.7 = 二维 DLA 维数，证明扩散受限聚集是视网膜血管形成底层过程（[ADS](https://ui.adsabs.harvard.edu/abs/1989PhyD...38...98F/abstract)）[V]
  - Schreiner & Buxbaum（1993）CCO 算法原始文献；Hamarneh & Jassi（2010）VascuSynth，*Computerized Medical Imaging and Graphics* 34(8):605-616（CCO 开源实现）[V]
  - Brown et al.（2024）"Physics-informed deep generative learning for quantitative assessment of the retina"，*Nature Communications* 15:6859，**DOI 10.1038/s41467-024-50911-y**——PI-GAN：生物物理规则（Murray 定律）生成动脉+静脉闭环血管树，仿真血流，CycleGAN 转眼底彩照风格；仅 100 张仿真图训练即近 SOTA 分割（DRIVE Dice 0.75）。（[Nature](https://preview-www.nature.com/articles/s41467-024-50911-y)）[V]
- **可行性**：中。纯 DLA/CCO 形态与真实 KW IV 有差距；Brown 方案"结构可仿真"价值大，但转风格用深度 CycleGAN——纯非深度版需退回"仿真树 + 手工/拼布纹理"。

### 1.4 补丁修复 inpainting（公共前置组件）
- **原理**：从真实图抠掉视盘/血管树/病变前景，得到干净背景库。
- **文献**：Criminisi, Pérez, Toyama（2004）"Region filling and object removal by exemplar-based image inpainting"，*IEEE TIP* 13(9):1200-1212，**DOI 10.1109/TIP.2004.833105**（[IEEE](https://ieeexplore.ieee.org/document/1323101)）[V]
- **可行性**：高。是 Fiorini、YoloCurvSeg 等管线的共同组件。

---

## 2. 结构/条件驱动生成（深度范式，作"结构先行"证据链引用）

这部分虽为深度方法，但证明了"**结构图先行 + 上色/转图**"范式在眼底上是主流且有效，是非深度组合获取结构 mask 的现成答案（直接跑分割器提取）。**本目录不实现这些深度方法**，只借鉴其结构解耦思路。

| 文献 | 方法 | 与本任务的相关点 |
|---|---|---|
| **Costa et al. 2018**，"End-to-End Adversarial Retinal Image Synthesis"，*IEEE TMI* 37(3):781-791，**DOI 10.1109/TMI.2017.2759102**（[PubMed](https://pubmed.ncbi.nlm.nih.gov/28981409/)）[V] | 对抗自编码器生成血管树 + GAN 上色 | 明确报告生成图与训练集"substantially different"但仍解剖一致 |
| **Zhao, Li, Cheng**（2017-2018）"Synthesizing Filamentary Structured Images with GANs"，**arXiv:1706.02185**（期刊版 MedIA 2018）[V] | FilaGAN/Tub-GAN，条件 GAN | 10-20 张训练即可合成眼底/神经元纤维结构 |
| **Alimanov & Islam**（2023）ICCP，**DOI 10.1109/ICCP56744.2023.10233841**（[arXiv:2308.08339](https://ar5iv.labs.arxiv.org/html/2308.08339)）[V] | 两阶段 DDPM：先生成血管树 → 条件转眼底图 | 结构→图像解耦的两阶段范式 |
| **Go et al.**（2024）CVPRW DCAMI（[链接](https://www.openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Go_Generation_of_Structurally_Realistic_Retinal_Fundus_Images_with_Diffusion_Models_CVPRW_2024_paper.html)）[V] | 无条件生成动/静脉 mask → pix2pixHD 转眼底 → 超分 | 临床图灵测试：眼科医生 >50% 判错 |
| **Fhima et al.**（2025）**arXiv:2503.01190**（[链接](https://ar5iv.labs.arxiv.org/html/2503.01190)）[V] | RLAD：潜在扩散 + 条件化血管/视盘视杯/病变多布局 | 多结构解耦，下游分割增益最高 +8.1% |
| **He et al.**（2023）"YoloCurvSeg"，*Medical Image Analysis*，**arXiv:2212.05566**（[链接](https://ar5iv.labs.arxiv.org/html/2212.05566)）[V] | 空间殖民算法生成血管曲线 mask + inpainting 抠背景 + 补丁级对比学习融合 | **非深度血管生成 + 非深度抠背景 + 深度融合**的混合管线，与本项目高度同构 |

---

## 3. 新奇组合头脑风暴（按文献基础分级）

### A. 血管骨架 mask + 补丁纹理填充 —— 文献基础：强
- **原理**：非深度方式（DLA/CCO/ASM/现成分割器）生成全新血管骨架 → 从 330 张真实图无病变区采集纹理贴片沿骨架拼布 → inpainting 修补缝隙。
- **支撑**：Fiorini 2014（血管纹理 quilting）、Criminisi 2004（inpainting）、He 2023（空间殖民+背景库+融合）、Efros & Freeman 2001。
- **可行性**：高。骨架全新保证"不重复"，纹理来自真实库保证"相似"，最贴合任务。

### B. PCA/特征空间 + 结构化采样 + 逆映射 —— 文献基础：中
- **原理**：对 330 张做 PCA，在系数空间做结构化采样（椭圆约束/插值/混排），逆映射出新图。
- **支撑（成分级）**：ASM 的 PCA 形状参数高斯采样（Bonaldi 2016）；稀疏字典共享系数重建；Costa 2018 潜在空间语义插值。
- **可行性**：中。整图 PCA 拉向"平均脸"，局部高频病灶需分块处理；系数采样距离需标定，否则易复制某张训练图。

### C. ⭐ 泊松融合/图像拼接制造"新"眼底图 —— 文献基础：强（已有眼底专版）
- **原理**：从真实图裁剪病变/血管/视盘区域，按临床规则用泊松无缝融合贴到另一张眼底图。
- **支撑**：Pérez, Gangnet, Blake（2003）"Poisson image editing"（SIGGRAPH）；**Yu et al.**（2021）"Multiple Lesions Insertion: boosting diabetic retinopathy screening through Poisson editing"，*Biomedical Optics Express* 12(5):2773-2789，**DOI 10.1364/BOE.420776**（[PubMed](https://pubmed.ncbi.nlm.nih.gov/34123503/)）[V]——健康眼底图上按 DR 分级规则插入渗出/出血/微动脉瘤模板，无黑斑无接缝，DR 筛查中优于过采样/裁剪/旋转等传统增强。
- **可行性**：**高，且极其契合本场景**。330 张全为 KW IV = 自带病变模板库，可做"病变跨图重排"：从图 A 取渗出簇贴到图 B、从图 C 取出血斑贴到图 D……直接产出新严重 DR 图。**注意**：病变模板必须是"跨图源"采样，否则等于复制原图局部。

### D. 多源补丁词典 + 稀疏组合 —— 文献基础：强（医学合成有专文）
- **原理**：从 330 张建 patch 词典（聚类/分类），新图 = 稀疏系数组合词典原子重建。
- **支撑**：字典学习用于医学图像合成是 2010s 早期正式方向（跨模态合成框架）；Huang et al. 几何正则联合词典（[Manchester](https://research.manchester.ac.uk/en/publications/geometry-regularized-joint-dictionary-learning-for-cross-modality/)）[V]；Fiorini 2014 的 30 万贴片 + K-means 4 簇词典是眼底实践版；K-SVD 为经典算法。
- **可行性**：中高。稀疏重建天然倾向回归训练 patch，需加多样性约束（原子选择熵正则、最近邻距离惩罚）。

### E. 血管树形变（统计形状模型 on 血管拓扑）+ 纹理合成 —— 文献基础：中
- **原理**：统计血管树的拓扑/几何分布，对树做形变（主模式采样/插值）出新树，再接纹理合成。
- **支撑**：Bonaldi 2016（ASM 形状参数采样）；**PartVessel**（MICCAI 2025，**arXiv:2507.15223**）[V]"Hierarchical Part-based Generative Model for Realistic 3D Blood Vessel"——RVAE 生成关键图（拓扑）+ Transformer 生成血管段（几何）+ 分层装配，明确"**拓扑与几何分离**"范式。
- **可行性**：中。PartVessel 是 3D 非眼底，迁移到 2D 需改造，但"拓扑分离几何"思路可直接借用。

### F. ⭐ 颜色/照度分离 + 独立重采样（Retinex 分解后重组合） —— 文献基础：强（眼底增强已用）
- **原理**：Retinex 把图分解为光照 L × 反射 R，独立扰动/跨图交换光照，保留反射，重乘出新图——照度变了、结构/病变语义不变。
- **支撑**：**Zhang, Li, Shin**（2022）"Robust color medical image segmentation on unseen domain by randomized illumination enhancement"，*Computers in Biology and Medicine* 145:105427，**DOI 10.1016/j.compbiomed.2022.105427**（[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482522002190)）[V]——无监督 Retinex 分解网络（ID-Net）把眼底图分解为反射+光照 → 随机化光照 → 重组合，视盘/视杯分割比 SOTA SDG 高 9.6% Dice。其思想就是"分解-重采样-重组合"。
- **可行性**：**高**。直接可做"跨图交换光照 + 保留各自反射"：同病级图之间交换颜色/照度几乎不改诊断语义，完全不需要重训网络。

### G. 其余待验证组合（无直接文献，标注需实验验证）
- **血管树形变 + 病变重分布**：E 的新骨架 + C 的病变模板跨图重排（思路可拼，需实验验证）。
- **几何形变 + 病变语义保持**：薄板样条/弹性形变作用整图，DR 病变大体保持（需实验验证；防变形过猛导致病灶断裂）。
- **频域重排**：交换/抖动 330 张图的相位谱，振幅保留（需实验验证）。

---

## 4. "相似但不相同"核心难点：小样本下的记忆/复制问题

**结论**：330 张对深度生成器处于记忆高风险区，非深度方法在"整体新"上有天然优势，但存在"局部复制"陷阱。

有文献支撑的经典证据链：
- **Feng et al.**（ICCV 2021）"When do GANs Replicate? On the Choice of Dataset Size"——**复制率随数据集规模指数衰减**；图像质量对规模呈 U 型；给出"单次估计最小安全数据量"工具。330 张远低于 BigGAN/StyleGAN 安全线。（[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_When_Do_GANs_Replicate_On_the_Choice_of_Dataset_Size_ICCV_2021_paper.pdf)）[V]
- **arXiv:1901.03396** "Detecting Overfitting of Deep Generative Networks via Latent Recovery"——GLO 在 128 张训练图开始记忆、约 8192 张才停止；**标准指标 FID 测不出记忆**，需 latent recovery 统计检验。[V]
- **Bhattacharjee et al.** **arXiv:2302.13181**——"data-copying"逐点定义与检测算法；全局距离检验会漏报（40% 输出为精确复制也可能被平均掉）。[V]

非深度方法的两面性：
- **优势**：补丁/过程化/重组合方法的"整体"（骨架、布局、光照、病变分布）由参数和过程决定，整图天然全新——不存在"整图复制"问题。
- **陷阱**：**局部复制**。拼布可见重复贴片/接缝，病变模板直贴等于复制原图局部。Efros & Freeman 原文自嘲"尽可能抄袭源图，再掩盖证据"。

**建议配套检测三件套**（对任何生成方法统一跑）：
1. 生成图 ↔ 330 训练图的**最近邻 SSIM/LPIPS 距离分布**
2. **全图级复制率**（Feng 2021 方法）
3. **补丁级重复块占比**（patch 自相似性）

任一生效阈值内都应淘汰/惩罚该生成器。

---

## 5. 面向 330 张 KW IV 的行动建议

1. **优先级**：**C（泊松病变重排）≈ F（Retinex 光照交换）> A（骨架+纹理拼布）> D（补丁词典稀疏组合）> B/E（PCA/拓扑形变）**。前两者有眼底专用文献背书、几乎不重训模型、产出天然"相似但不相同"。
2. **结构图来源**：直接跑现成眼底分割器给 330 张出血管 mask + 病变 mask + 视盘/视杯 mask，再走非深度组合；或参考 He 2023 的"分割 + inpainting 抠净背景库 + 生成 mask"流程。
3. **记忆检测必配**：复制率 / NN 距离 / 重复块占比三件套，否则任何方法都可能在 330 张下悄悄复制。
4. **利用"全为严重 DR"特性**：病变模板库 = 全部 330 张的渗出/出血/微动脉瘤/棉絮斑；MLI 式"跨图病变重排"可将"病种分布、病灶密度、位置"显式控制为新变量，这在深度无条件生成器里难以显式控制。
