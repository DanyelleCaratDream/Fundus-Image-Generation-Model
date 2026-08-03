# 文献综述①：传统 ML 图像生成方法盘点（18 法）

> 调研目的：建立"非深度学习图像生成"的方法谱系，评估每种方法对 330 张严重 DR（KW IV 级）眼底图"相似但不相同"生成的可行性。
> 边界：只盘传统 ML / 经典 CV 方法；深度生成与预训练见 `00-README.md` 边界表。
> 验证说明：每条文献均经 WebSearch 逐条核实（含卷期页码/DOI/arXiv 号），来源标注 [V]（已验证）。
> 更新日期：2026-08-03

---

## 结论先行

18 个方法按可行性分三档：

| 档位 | 方法 | 一句话理由 |
|:--|:--|:--|
| **高** | Image Quilting 拼布、眼底补丁合成管线、稀疏字典、Poisson 融合、颜色/直方图匹配 | 有小样本眼底先例或医学证据，保纹理不保全局 |
| **中** | COCA（Copula PCA）、GMM（紧凑表示上）、AAM/SSM、exemplar inpainting、Image Analogies | 可用但需组合/标注，不能单独生成 |
| **低** | 纯 PCA/PPCA/FA、KDE、FRAME、纯 MRF | 过平滑/无宏观结构，仅作否定性对照 |

---

## 1. 线性 / 因子模型（Latent Linear Models）

### 1.1 PCA 重建 / 特征脸（Eigenpictures / Eigenfaces）
- **原理**：图像拉成向量 → 用训练集主成分的线性组合近似重建 → 扰动/插值主成分系数即生成"新"图。
- **文献**：
  - Sirovich & Kirby, *Low-dimensional procedure for the characterization of human faces*, JOSA A 4(3):519–524, 1987. DOI [10.1364/JOSAA.4.000519](https://opg.optica.org/josaa/abstract.cfm?URI=josaa-4-3-519) [V]
  - Turk & Pentland, *Eigenfaces for Recognition*, J. Cognitive Neuroscience 3(1):71–86, 1991. DOI [10.1162/JOCN.1991.3.1.71](https://direct.mit.edu/jocn/article/3/1/71/3025/Eigenfaces-for-Recognition) [V]
- **330 张可行性**：**中**（作插值）。致命弱点：严重 DR 病灶（出血/渗出/微动脉瘤）在像素空间不是线性子空间，PCA 生成不出新病灶形态；非高斯边缘产生"模糊叠加"伪影（Copula Eigenfaces 正是为此提出，见 §6.1）。

### 1.2 概率 PCA（PPCA）
- **原理**：把 PCA 写成高斯隐变量生成模型 x = Wz + μ + ε, z~N(0,I)，EM 估计后可从隐空间采样。
- **文献**：Tipping & Bishop, *Probabilistic Principal Component Analysis*, JRSS-B 61(3):611–622, 1999. DOI [10.1111/1467-9868.00196](https://academic.oup.com/jrsssb/article-abstract/61/3/611/7083217) [V]
- **330 张可行性**：**低–中**。提供显式噪声 ε 作天然"多样性注入"，但同样受线性限制，宜作 baseline 或与 Copula/字典组合。

### 1.3 因子分析（Factor Analysis）
- **原理**：FA 与 PCA 同属线性高斯生成模型，区别是每个隐变量可有独立噪声方差。PCA/FA/PPCA/ICA 统一在"从隐空间采样→线性映射→加噪"框架下。
- **文献**：Roweis & Ghahramani, *A Unifying Review of Linear Gaussian Models*, Neural Computation 11(2):305–345, 1999. DOI [10.1162/089976699300016674](https://direct.mit.edu/neco/article-abstract/11/2/305/6249/A-Unifying-Review-of-Linear-Gaussian-Models) [V]
- **330 张可行性**：**低**。与 PCA 同类，仅作对照 baseline。

---

## 2. 混合模型 / 密度估计

### 2.1 高斯混合模型（GMM）
- **原理**：p(x)=Σπₖ𝒩ₖ(x)，"选分量→从高斯采样"两步生成新样本；可条件于标签图或低维表示。
- **文献**：
  - Yang & Chakraborty, *A GMM based algorithm to generate point-cloud and its application to neuroimaging*, arXiv:[1911.01705](http://arxiv.org/abs/1911.01705), 2019 [V]
  - Gepperth & Pfülb, *Image Modeling with Deep Convolutional Gaussian Mixture Models*, arXiv:[2104.12686](https://ar5iv.labs.arxiv.org/html/2104.12686), 2021 [V]
  - Billot 等, *lab2im*（条件于标签图采样 GMM 生成脑 MRI，SynthSeg 的数据合成器），[GitHub](https://github.com/fracogno/lab2im) [V]
- **330 张可行性**：**中（紧凑表示上）**。不能直接对像素建模，必须先降维（补丁/PCA 潜空间/结构参数）。可做"健康背景补丁 GMM + 病灶补丁字典"两段式。

### 2.2 核密度估计（KDE）
- **原理**：Parzen 窗核密度估计像素/补丁分布，从该密度采样生成。
- **文献**：Sinha & Gupta, *A Fast Nonparametric Noncausal MRF-Based Texture Synthesis Scheme Using a Novel FKDE Algorithm*, IEEE TIP 19(3):561–572, 2010. DOI [10.1109/TIP.2009.2036685](https://pubmed.ncbi.nlm.nih.gov/19933004/) [V]
- **330 张可行性**：**低–中**。本质是"平滑的经验分布"，采样高度复制已有模式、新颖性有限；严重 DR 病灶作为稀有模式会被平均化。

### 2.3 FRAME（滤波 + 随机场 + 最大熵）
- **原理**：滤波响应直方图作约束，最大熵导出 Gibbs 随机场分布，Gibbs 采样合成纹理。
- **文献**：Zhu, Wu & Mumford, *Filters, Random Fields and Maximum Entropy (FRAME)*, IJCV 27(2):107–126, 1998. DOI [10.1023/A:1007925832420](https://dl.acm.org/doi/abs/10.1023/A:1007925832420) [V]
- **330 张可行性**：**低**。纹理级模型，无法保持眼底宏观解剖（血管走向/视盘/病灶布局）。

---

## 3. 纹理合成 / 补丁方法（★ 本方向主干）

### 3.1 Efros–Leung 非参数纹理合成
- **原理**：把图像当 n-gram，逐像素在样本图中搜索相似邻域、按条件分布随机取像素。
- **文献**：Efros & Leung, *Texture Synthesis by Non-parametric Sampling*, ICCV 1999. [项目页](http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html) / [GitHub 实现](https://github.com/goldbema/TextureSynthesis) [V]
- **330 张可行性**：**中**（眼底背景/无血管区）。有医学先例（Fiorini 引入眼底合成）。

### 3.2 Image Quilting 图像拼缝（含纹理迁移）
- **原理**：从样本图切补丁，SSD+误差容差选块、min-cut 找接缝拼接成新图；加 correspondence map 即"纹理迁移"。
- **文献**：Efros & Freeman, *Image Quilting for Texture Synthesis and Transfer*, SIGGRAPH 2001, pp. 341–346. DOI [10.1145/383259.383296](https://dl.acm.org/doi/10.1145/383259.383296) [V]
- **330 张可行性**：**中–高**。补丁法中医学验证最扎实的路线（见 3.3）。注意：输出是训练像素重组，需配合去重/多样性筛选防复制。

### 3.3 ⭐ 眼底直接先例：Fiorini 补丁式合成 + Magnusson 分割训练
- **原理**：patch-based 拼贴（背景/黄斑，image quilting + k-means 补丁字典 + min-cut 接缝）+ 参数模型（视盘/血管树）混合管线，产出带完整 GT 的合成眼底图。
- **文献**：
  - Fiorini, De Biasi, Ballerini, Trucco & Ruggeri, *Automatic Generation of Synthetic Retinal Fundus Images*, STAG 2014. DOI [10.2312/stag.20141238.041-044](https://dlold.eg.org/handle/10.2312/stag.20141238.041-044) [V]
  - Magnusson, Afifi, Zhang, Ley & Hellwich, *Synthesizing Fundus Photographies for Training Segmentation Networks*, DeLTA 2021, pp. 67–78. DOI [10.5220/0010618100670078](https://www.scitepress.org/PublishedPapers/2021/106181/)（[开源](https://github.com/jannessm/RetinaSynthesis)）[V]
- **330 张可行性**：**高**。方法为小样本而生（Fiorini 仅用 15–45 张 HRF 图），完全适配 330 张严重 DR 图。**这是"传统方法合成眼底图增强下游任务"的最强证据。**

### 3.4 稀疏编码 / 字典学习
- **原理**：图像补丁稀疏表示为过完备字典少数原子的线性组合，重采样稀疏系数生成新补丁组合。
- **文献**：Yang, Wright, Huang & Ma, *Image Super-resolution via Sparse Representation*, IEEE TIP 19(11):2861–2873, 2010. DOI [10.1109/TIP.2010.2050625](https://ui.adsabs.harvard.edu/abs/2010ITIP...19.2861Y/exportcitation) [V]
- **330 张可行性**：**中**。字典从 330 张图补丁学得，可生成"新组合"病灶/纹理补丁；全局布局仍需拼接编排，严重 DR 病灶字典需人工裁剪标注。

### 3.5 示例式修补（Exemplar-based Inpainting）
- **原理**：按"置信度×图像梯度"优先级，用最相似补丁填充待修复区域（传播纹理与结构）。
- **文献**：Criminisi, Pérez & Toyama, *Region Filling and Object Removal by Exemplar-based Image Inpainting*, IEEE TIP 13(9):1200–1212, 2004. DOI [10.1109/TIP.2004.833105](https://ieeexplore.ieee.org/document/1323101) [V]
- **330 张可行性**：**中–高**。与"健康底版 + 病灶补丁"范式天然契合（正常眼底图上"填充"病灶区域可精确控制 DR 严重度）。

---

## 4. 统计形状 / 外观模型（ASM / AAM / SSM）

### 4.1 主动外观模型 AAM
- **原理**：PCA 联合建模 landmarks 形状与纹理统计，模型参数直接合成完整外观图像——**本质就是可采样的生成式外观模型**。
- **文献**：Cootes, Edwards & Taylor, *Active Appearance Models*, IEEE TPAMI 23(6):681–685, 2001. DOI [10.1109/34.927467](https://research.manchester.ac.uk/en/publications/active-appearance-models) [V]
- **330 张可行性**：**中**。两个瓶颈：严重 DR 病灶在像素空间非高斯线性（AAM 表达有限）；需逐图 landmark 标注（330 张成本高）。若只用少量结构 landmark（视盘+血管主干）做对齐插值则可行。

### 4.2 ASM / SSM 在眼底的应用
- **原理**：SSM 用 PCA 建模形状点集变化，ASM 以该形状先验驱动分割/重建；眼底上用于视盘、盘周萎缩（PPA）建模。
- **文献**：
  - Li et al., *Peripapillary Atrophy Segmentation Based on ASM Loss*, IEEE ISBI 2022. DOI [10.1109/ISBI52829.2022.9761687](https://ieeexplore.ieee.org/document/9761687) [V]
  - *Semi-supervised peripapillary atrophy segmentation with shape constraint*, Computers in Biology and Medicine, 2023. [链接](https://www.sciencedirect.com/science/article/abs/pii/S0010482523009290) [V]
- **330 张可行性**：**中**（辅助）。本身是"形状先验"而非图像生成器，可用其 PCA 形状空间做结构插值（新视盘/血管走向），再与补丁法叠加。

---

## 5. 图像类比 / 风格迁移经典方法

### 5.1 Image Analogies 图像类比
- **原理**：从 (A,A') 学"滤波"，多尺度自回归应用到新图 B，得到 "B' : B = A' : A"。
- **文献**：Hertzmann, Jacobs, Oliver, Curless & Salesin, *Image Analogies*, SIGGRAPH 2001, pp. 327–340. DOI [10.1145/383259.383295](https://mrl.cs.nyu.edu/publications/image-analogies/) [V]
- **330 张可行性**：**中**。可用"健康→病灶"映射增强多样性，但需成对训练图（眼底难严格配对），宜作辅助。

### 5.2 Color Transfer 颜色迁移（Reinhard）
- **原理**：lαβ（去相关）空间匹配源/参考图均值与标准差，把参考图色彩统计施加到目标图。
- **文献**：Reinhard, Ashikhmin, Gooch & Shirley, *Color transfer between images*, IEEE Computer Graphics and Applications 21(5):34–41, 2001. DOI [10.1109/38.946629](https://research-information.bris.ac.uk/en/publications/color-transfer-between-images/) [V]
- **330 张可行性**：**高**（多样性增强组件）。注入每张图颜色/光照变化、不影响病灶结构；需 ROI 裁剪/血管分离适配（黑边框干扰）。

### 5.3 直方图匹配 / 强度标准化（医学）
- **原理**：直方图 landmark（分位数/众数）分段线性映射，把图像强度/颜色分布对齐到参考分布。
- **文献**：Nyúl & Udupa, *On standardizing the MR image intensity scale*, Magnetic Resonance in Medicine 42(6):1072–1081, 1999. DOI [10.1002/(SICI)1522-2594(199912)42:6<1072::AID-MRM11>3.0.CO;2-M](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291522-2594%28199912%2942%3A6%3C1072%3A%3AAID-MRM11%3E3.0.CO%3B2-M) [V]
- **330 张可行性**：**高**。眼底预处理综述指出**直方图匹配（核平滑版）是唯一显著改善 DR 病灶检测的颜色归一化方法**，优于 Reinhard [V]。

---

## 6. 其他可试方法

### 6.1 ⭐ Copula 依赖建模（高斯 Copula + PCA / COCA）
- **原理**：高斯 copula 把"依赖结构"与"边缘分布"解耦——rank 变换到高斯潜空间做 PCA，再按训练集经验边缘分布逆变换采样。**解决"PCA 假设高斯导致合成伪影"。**
- **文献**：Egger, Kaufmann, Schönborn, Roth & Vetter, *Copula Eigenfaces: Semiparametric Principal Component Analysis for Facial Appearance Modeling*, VISIGRAPP 2016 (GRAPP), pp. 50–58. DOI [10.5220/0005718800480056](https://www.scitepress.org/Papers/2016/57188/) [V]
- **330 张可行性**：**中–高**。眼底分布明显非高斯（黑背景/亮视盘/暗病灶），COCA 是修正"线性模型生成伪影"的最直接手段，是 §1 线性方法与"非高斯修正"结合最有前景的组合。

### 6.2 马尔可夫随机场（MRF / GRMF）
- **原理**：图像建模为局部邻域条件分布（Gibbs/二项分布），从条件分布采样生成。
- **文献**：Cross & Jain, *Markov Random Field Texture Models*, IEEE TPAMI 5(1):25–39, 1983. DOI [10.1109/TPAMI.1983.4767341](https://pubmed.ncbi.nlm.nih.gov/21869080/) [V]
- **330 张可行性**：**低**。纯 MRF 依赖小邻域、无法保持宏观解剖；对背景纹理可行，作补丁级多样化辅助。

### 6.3 ⭐ Poisson Image Editing 泊松图像编辑（融合组件）
- **原理**：求解泊松方程，让插入区域保持自身梯度并与背景无缝融合——**不是生成器，而是把病灶补丁/新结构嵌进合成图的必备融合工具**。
- **文献**：Pérez, Gangnet & Blake, *Poisson Image Editing*, ACM ToG 22(3):313–318 (SIGGRAPH 2003). DOI [10.1145/1201775.882269](https://www.inf.ed.ac.uk/publications/report/1094.html) [V]
- **330 张可行性**：**高**（组件）。与 quilting/字典/inpainting 组合成"健康底版 + 病灶补丁 + 泊松融合"完整管线（眼底增强有专门验证，见 02 篇 3.2）。

---

## 综合：对本项目的可行性排序

| 排名 | 方法 | 可行性 | 一句话理由 |
|---|---|---|---|
| 1 | **补丁拼贴/合成管线**（quilting + 稀疏字典 + Poisson） | **高** | Fiorini/Magnusson 在 15–45 张眼底图上验证并提升分割 SOTA |
| 2 | **COCA（Copula 修正 PCA 潜空间采样）** | 中–高 | 解决眼底非高斯导致的线性伪影 |
| 3 | **颜色/直方图匹配多样性增强** | 高 | 成本极低，医学证据显示直方图匹配对 DR 检测增益明确 |
| 4 | **GMM 于紧凑表示**（补丁/参数/潜空间） | 中 | 直接对像素建 GMM 必败；降维表示上采样可出"新组合" |
| 5 | **AAM/SSM 结构插值** | 中 | 生成解剖合理新结构，但需 landmark、对病灶形态表达有限 |
| 6 | **KDE / FRAME / 纯 MRF** | 低–中 | 纹理级模型，无法保持眼底宏观解剖 |
| 7 | **纯 PCA/PPCA/FA 像素级生成** | 低–中 | 线性假设 + 样本量限制，过平滑、无新病灶形态 |

**关键风险**：补丁/拼贴类输出是训练像素重组，有"复制训练图"（identity leakage）风险 → 生成后必须做最近邻去重（L2/LPIPS 距离筛选），确保"相似但不相同"（检测方法见 02 篇 §4）。
