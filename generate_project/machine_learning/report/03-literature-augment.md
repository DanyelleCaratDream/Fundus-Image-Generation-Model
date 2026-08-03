# 文献综述③：小样本医学图像经典数据增广证据（传统 ML 范畴）

> 调研目的：盘点**非生成式经典增广**（传统 ML / 经典 CV 手段）对"扩充分类器训练集"的实证效果。
> 边界声明：本目录只做传统 ML。**已做的深度生成方法**（VAE/DCGAN/DDPM）评估见 `research-report/evaluation_report.md`；**预训练/迁移学习**相关文献见 `generate_project/pretrained/`。此处不重复。
> 验证说明：所有引用经网络检索验证存在；无编造。
> 更新日期：2026-08-03

---

## 结论先行

**对小样本医学图像分类，经典几何/光度增广与 Mixup 是证据最强、性价比最高的增广手段。** 经典增广（旋转、翻转、缩放、弹性形变、CLAHE、mixup 等）全部是传统 ML / 经典 CV 手段——零训练成本、标签保持性好。

> 深度生成模型扩充（GAN/扩散）与经典增广的直接对比结论见 `generate_project/pretrained/`（生成模型扩充重定位讨论）。一句话带过：文献普遍显示生成图扩充的判别力提升 ≤ 经典增广，C2 的价值锚定在"经典增广做不到的新病灶组合"而非"比增广强"。

---

## 一、方法排序（对"扩充分类器"的证据强度）

| 排名 | 方法 | 证据强度 | 对分类器增益预期 | 风险/成本 |
|:--|:--|:--|:--|:--|
| 1 | **经典几何+光度增广**（旋转±10°、水平翻转、缩放、轻度弹性形变、CLAHE/颜色扰动） | 最强（医学近 20 年积累 + 眼底专项） | 中-高：准确率/AUC 稳定 +1~4 点，标签保持性最好 | 极低；唯一坑是强度过大的光度变换 |
| 2 | **Mixup**（同类别样本凸组合；轻量 CutMix 次之） | 强（DR 专项 + 眼底/脑肿瘤基准） | 高：准确率提升同时改善鲁棒性/校准 | 低；无新网络；同类别插值不破坏标签 |
| 3 | **特征空间增广/噪声**（GNUS、特征空间插值、SMOTE 变体） | 中（表格/特征级证据，CNN 特征空间证据弱） | 中：对类不平衡帮助明确；对判别提升有限 | 中；CNN 特征空间插值可能产生离流形样本 |
| 4 | **学习式增广**（RandAugment 受限版；AutoAugment 需谨慎） | 中（医学图像证据混合，已知失败案例） | 中：限定几何+安全光度时可用；直接套用已知会掉点 | 中-高；需手动调参/限制变换集 |

---

## 二、逐主题展开

### 主题 1：经典几何/光度增广

**结论**：旋转（±10°）、水平翻转、缩放、轻度弹性形变与 CLAHE/颜色扰动，在眼底/DR 图像上证据最充分、最安全、零成本，组合使用稳定带来 +1~4 点准确率/AUC 提升；但强度过大的光度/极端透视变换已知有害。

代表文献（全部可验证）：
- **Simard, Steinkraus & Platt（2003）**，ICDAR 2003，DOI [10.1109/ICDAR.2003.1227801](https://doi.org/10.1109/ICDAR.2003.1227801) [V]——提出弹性形变（高斯平滑位移场，σ/α 控制强度），医学增广祖师级引用。
- **Ronneberger, Fischer & Brox（2015）**，MICCAI 2015，arXiv [1505.04597](https://arxiv.org/abs/1505.04597) [V]——U-Net 用激进弹性形变"以极少量标注图训练深度网络"成功。
- **Abraham et al.（2026）**，Digital Health（SAGE），DOI [10.1177/20552076261461391](https://doi.org/10.1177/20552076261461391) [V]——EfficientNet-B0 + 1200 张眼底图，6 种增广：总体 AUC 96.55%→97.23%、准确率 85.83%→89.58%；**CLAHE 对 DR 最优，旋转/颜色扰动对 AMD 最优——增广效果是病种特异的**。

**与本项目相关性（高）**：眼底图以黄斑为中心的**径向对称**结构使旋转在解剖上天然合理，水平翻转不改变 KW IV 病灶标签——这两类是最安全、最优先的增广。建议把 CLAHE 与轻度亮度/对比度扰动纳入管线（DR 专项证据支持），弹性形变强度调低，显式排除极端强度光度变换与强透视变换。

### 主题 2：Mixup / CutMix / 拼接类增广

**结论**：Mixup（同类别样本凸组合）在小样本医学图像上证据明确地提升准确率、鲁棒性与不确定性质量，对 DR 分级有专项支持，仅次于经典增广；CutMix 在眼底有增益但弱于 Mixup。

代表文献：
- **Zhang, Cisse, Dauphin & Lopez-Paz（2018）**，ICLR 2018，arXiv [1710.09412](https://arxiv.org/abs/1710.09412) [V]——mixup 基线方法，缓解过拟合/记忆化。
- **Yun et al.（2019）**，ICCV 2019，arXiv [1905.04899](https://arxiv.org/abs/1905.04899) [V]——CutMix。
- **Ahamed & Amireskandari（2025）**，MICCAI 2025 DEMI，arXiv [2508.14266](https://arxiv.org/abs/2508.14266)，DOI [10.1007/978-3-032-08009-7_22](https://link.springer.com/chapter/10.1007/978-3-032-08009-7_22) [V]——**DR 专项**：DDR 上 Mixup 对 ResNet-50/CoaT 均给最佳 top-1（83.8%/84.2%）且改善校准；**注意：同研究 CLAHE 反而损害模型确定性**（与主题 1 形成张力）。
- **MediAug（2025）**，MIUA 2025，arXiv [2504.18983](https://arxiv.org/abs/2504.18983) [V]——眼底眼病数据集：YOCO 对 ResNet-50 增益最大（91.60%）、CutMix 对 ViT-B 最大（97.94%）；脑肿瘤上 Mixup +2.79pp。

**与本项目相关性（高）**：本项目 330 张**全部属于同一严重类别**——Mixup 在"同一类别内插值"时标签天然保持，无需任何新网络。这是最被低估、最值得优先试的方案；应在 Mixup α 上做小网格搜索（α=0.2/0.4/1.0）。CutMix 拼接病灶贴片可能产生解剖不真实的复合图，建议轻量使用或仅做同类别拼接。

### 主题 3：特征空间增广 / SMOTE 变体 / 噪声注入

**结论**：噪声注入（GNUS）与 SMOTE 类方法在类不平衡场景有明确但有限的帮助（提升平衡类指标多于判别能力）；对极小的数据集（<100 样本）增益消失甚至无效；CNN 特征空间插值需谨慎。

代表文献：
- **Chawla, Bowyer, Hall & Kegelmeyer（2002）**，JAIR 16:321–357，DOI [10.1613/jair.953](https://www.jair.org/index.php/jair/article/view/10302) [V]——SMOTE 开山之作。
- **DeVries & Taylor（2018）**，arXiv [1702.05538](https://arxiv.org/abs/1702.05538) [V]——在学到特征空间加高斯噪声/插值/外推做增广。
- **Beinecke & Heider（2021）**，BioData Mining 14:49，DOI [10.1186/s13040-021-00283-6](https://doi.org/10.1186/s13040-021-00283-6) [V]——**关键负面发现：在最小数据集（72–100 样本）上增广根本不带来提升**——与本项目 330 张边界规模直接相关。

**与本项目相关性（中）**：DR 分级若是"严重 vs 非严重"二分类且严重类为少数类，SMOTE/GNUS 可缓解不平衡；但深网特征空间插值需谨慎。警惕 Beinecke 结论——330 张是小样本边界，需实证而非假定增广必然有效。

### 主题 4：学习式自动增广（RandAugment / AutoAugment）

**结论**：为自然图像设计的自动增广在医学小样本上证据混合——用对（限定几何+安全光度、手动调幅）可带来最高约 +17% 准确率提升；直接套用（尤其强度型变换）在 MRI 上已知导致 -10~-17pp 掉点，不能开箱即用。

代表文献：
- **Cubuk et al.（2019）**，CVPR 2019，arXiv [1805.09501](https://arxiv.org/abs/1805.09501) [V]——AutoAugment，RL 搜索，重计算。
- **Cubuk, Zoph, Shlens & Le（2020）**，arXiv [1909.13719](https://arxiv.org/abs/1909.13719) [V]——RandAugment，仅 2 超参（N 操作数、M 强度），医学小样本最常用起点。
- **Suzuki（2022）**，CVPR 2022（Oral），arXiv [2202.12513](https://arxiv.org/abs/2202.12513) [V]——TeachAugment，教师网络约束对抗增广不破坏语义。
- **Pattilachan et al.（2023）**，arXiv [2301.02181](https://arxiv.org/abs/2301.02181) [V]——**负面证据核心**：MRI 分类上 AutoAugment/RandAugment/AugMix 全部显著掉点（基线 61.7% vs 51.0~55.3%）；去掉强度类变换只留几何后回到基线——证明破坏源是强度型变换。

**与本项目相关性（中）**：建议只用 **RandAugment 的"几何+轻度光度"受限子集**（旋转/翻转/缩放/平移，禁用颜色反转、海报化、锐化过强等破坏病灶对比度的操作），M 幅度小范围网格搜索（M∈{5,10,15}）。避免直接把 AutoAugment 预训练策略搬过来。

### 主题 5：小样本训练的已知坑（增广不足/过度权衡、正则化）

**结论**：增广不足与过度都会伤害小样本训练——过度增广（尤其强度型）会破坏诊断信息并让模型学"伪影"而非"解剖"；增广必须与 dropout/weight decay/early stopping/压缩容量配套，二者互补；极小数据集增广可能无效。

代表文献：
- **Pattilachan et al.（2023）**，arXiv [2301.02181](https://arxiv.org/abs/2301.02181) [V]（见主题 4）。
- **CyclicAugment（2025）**，IEEE Access，[Article 11005973](https://ieeexplore.ieee.org/document/11005973) [V]——"过度增广会扭曲关键诊断信息、早训不稳定、增过拟合风险"；动态循环强度 +8.8% 准确率。
- **Sun et al.（2024）**，arXiv [2409.12355](https://arxiv.org/abs/2409.12355) [V]——小样本把增广与正则化显式组合实现 85–88% 准确率。
- **Beinecke & Heider（2021）**，DOI [10.1186/s13040-021-00283-6](https://doi.org/10.1186/s13040-021-00283-6) [V]：<100 样本增广可能无收益。
- 皮肤病变综述（2024），Multimedia Tools and Applications，DOI [10.1007/s11042-024-20145-7](https://doi.org/10.1007/s11042-024-20145-7) [V]：小样本+类不平衡下中等复杂度模型优于大模型。

**与本项目相关性（高）**：所有消融应在同一验证协议下做：**患者级划分、报告 AUC + 平衡准确率/MCC**，记录增广幅度与过拟合曲线（训练/验证 gap）。

---

## 三、给本项目的落地建议（按优先级）

1. **先建评估基线**：预训练骨干 + 患者级划分 + 经典增广（旋转±10°、水平翻转、缩放、轻度弹性、CLAHE/轻亮度-对比度）→ 记录 AUC/平衡准确率。此即所有对比的参照。（注：预训练骨干选择见 `generate_project/pretrained/`）
2. **加入 Mixup（α≈0.2–0.4 网格）**：同类（KW IV）间插值，预期稳定增益，且改善校准。
3. **RandAugment 受限子集**（仅几何+轻度光度，M 小幅度网格）作第二候选；禁强度型变换。
4. **配套正则化**：dropout/weight decay/early stopping/降容量，与增广并列执行。
5. **评估纪律**：患者级（不是图像级）划分；报告 AUC + 平衡指标 + 校准；警惕 simplicity bias。

---

## 四、参考文献汇总（全部经检索验证）

1. Simard, Steinkraus, Platt. ICDAR 2003. DOI: [10.1109/ICDAR.2003.1227801](https://doi.org/10.1109/ICDAR.2003.1227801)
2. Ronneberger, Fischer, Brox. MICCAI 2015. arXiv: [1505.04597](https://arxiv.org/abs/1505.04597)
3. Abraham et al. Digital Health, 2026. DOI: [10.1177/20552076261461391](https://doi.org/10.1177/20552076261461391)
4. Zhang et al. ICLR 2018. arXiv: [1710.09412](https://arxiv.org/abs/1710.09412)
5. Yun et al. ICCV 2019. arXiv: [1905.04899](https://arxiv.org/abs/1905.04899)
6. Ahamed & Amireskandari. MICCAI 2025 DEMI. arXiv: [2508.14266](https://arxiv.org/abs/2508.14266)
7. MediAug. MIUA 2025. arXiv: [2504.18983](https://arxiv.org/abs/2504.18983)
8. Chawla et al. JAIR 16:321–357, 2002. DOI: [10.1613/jair.953](https://www.jair.org/index.php/jair/article/view/10302)
9. DeVries, Taylor. arXiv: [1702.05538](https://arxiv.org/abs/1702.05538)
10. Beinecke, Heider. BioData Mining 14:49, 2021. DOI: [10.1186/s13040-021-00283-6](https://doi.org/10.1186/s13040-021-00283-6)
11. Cubuk et al. CVPR 2019. arXiv: [1805.09501](https://arxiv.org/abs/1805.09501)
12. Cubuk, Zoph, Shlens, Le. arXiv: [1909.13719](https://arxiv.org/abs/1909.13719)
13. Suzuki. CVPR 2022. arXiv: [2202.12513](https://arxiv.org/abs/2202.12513)
14. Pattilachan et al. arXiv: [2301.02181](https://arxiv.org/abs/2301.02181)
15. CyclicAugment. IEEE Access 2025. [IEEE 11005973](https://ieeexplore.ieee.org/document/11005973)
16. Sun et al. arXiv: [2409.12355](https://arxiv.org/abs/2409.12355)
17. 皮肤病变增广综述. Multimedia Tools and Applications, 2024. DOI: [10.1007/s11042-024-20145-7](https://doi.org/10.1007/s11042-024-20145-7)
