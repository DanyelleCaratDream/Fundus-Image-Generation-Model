# 质量提升方法清单（生成细节升级线）

> 回答"loss 平台期之后，细节质量还有哪些提升方法"。
> 来源：2026-10-17 六路文献检索（arXiv / PMC / ScienceDirect / Springer / 中文平台）＋ [报告 07 预训练调研](../generate_project/machine_learning/report/07-pretrained-gen-models.md)（2026-08-26，8GB 可行性已核验）合并。
> 背景：D3（DL-EX-021）末段 loss 0.049~0.065 平台，与成功配方 0.052 同量级——L1 噪声预测存在信息论下限，**细节提升不在损失侧**，在采样端/数据端/后处理端/预训练先验。与论文 6.1（信息量瓶颈）自洽。
> 优先级：**A1+A2（零训练，先做）→ C（SR→256 既定路线）→ B1（LoRA，真正抬天花板的下一实验线）**。

---

## A. 零训练（立即可做）

### A1 SDEdit 自精修 —— 已预注册 DL-EX-023（等 D4 完）

- **做法**：对生成图加噪到中等噪声（t≈300~400），用**同一个模型**去噪回来：高频伪影（白点、边缘噪点）被重采样，结构保留。
- **依据**：SDEdit（Meng et al., ICLR 2022, [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)）；Stable Diffusion 的 img2img 即同源机制。
- **成本**：一次推理秒级，零训练。

### A2 自动质量筛选器（"生成+筛选"管线的自动化）

- **先例**：
  - ECC_DM_for_DR（MICCAI 2025，[GitHub](https://github.com/AlanZhang1995/ECC_DM_for_DR)）：按 DR 级生成 + 分类器语义筛选，DDR 平衡准确率 66.84→74.20（训练需 24GB，**方法借鉴、方案不照搬**）
  - model-guided filtering（[Computerized Medical Imaging and Graphics 2026](https://www.sciencedirect.com/science/article/pii/S0895611126000789)）：选择器对生成图打分丢弃劣质样本，**"curated 优于更大未筛选集"**，下游 Dice 0.24→0.44（别人的数字，引用时注明）
  - SIF 合成图过滤器（[arXiv:2609.35226](https://arxiv.org/html/2609.35226v1)）
- **我们的做法**：用人工标注过的干净/失败样本训一个轻量二分类器（或先 BRISQUE 排序 top-k），替代手工挑图（干净率 25~30% 的口子）。
- **论文衔接**：写"采用/借鉴该范式"（引用红线：禁"首创筛选管线"）。

### A3 EMA 重塑（白捡）

- NVIDIA 2025 的做法：训练后用已有 checkpoint **重建不同长度的 EMA**（[Rethinking how to train diffusion models](https://developer.nvidia.cn/blog/rethinking-how-to-train-diffusion-models/)）——不重训；EMA 0.9999 偏"长"时重建短 EMA 有时更锐。
- 与 D3 的 raw/EMA 之谜（DL-EX-021 双轨判定）正好配套试。

## B. 低成本训练（8GB 内）

### B1 SD1.5/2.1 + LoRA + 血管 ControlNet（真正抬天花板的候选）

- **报告 07 已核验**：Kohya 官方口径 LoRA 训练 6GB 起，8GB 卡可跑；现成眼底套件 [KylianSu/vessel-bezier-retinal-weights](https://huggingface.co/KylianSu/vessel-bezier-retinal-weights)（SD2.1 + LoRA rank64 + 血管 ControlNet + restore 脚本）。
- **为什么能抬天花板**：预训练先验 = 外来信息量，正面回应 6.1 的"容量/轮数不可替代信息量"——330/193 张学不出的血管拓扑先验，SD 在数十亿自然图上学过。
- **新文献支持**：[MeLo](https://arxiv.org/html/2311.08236v2)（医学域 LoRA 优于全量微调）、[SeLoRA](https://arxiv.org/html/2408.07196v1)、[SD 微调范式系统评测](https://arxiv.org/html/2508.14931v1)。
- **定位**：D3/D4 判定 + 5.6 占位回填之后的**下一实验线**，预注册后启动。

### B2 损失增量（局部问题，抬不了天花板）

多尺度 LPIPS / 边缘感知损失 / 频域损失（REPORT §5.11.3 已列）——只解决白点/噪点类局部缺陷，优先级放低。

### B3 序数条件

[berenslab/OrdinalDiffusionModels](https://github.com/berenslab/OrdinalDiffusionModels)：DR 0–4 级做序数条件（EyePACS QWK 0.79→0.87），比 one-hot 贴合分级本质；需自训。

## C. 眼底特化增强/SR（接生成输出 → 256）

| 方案 | 出处 | 说明 |
|:--|:--|:--|
| FD3 扩散桥眼底增强 | [arXiv:2409.12377](https://arxiv.org/html/2409.12377v1) | 应对宽范围退化 |
| ESDiff 增强+血管分割联合 | [PMC10898574](https://pmc.ncbi.nlm.nih.gov/articles/PMC10898574/) | 低质量视网膜图增强 |
| 通用医学扩散增强 | [Nature Commun Med 2025](https://www.nature.com/articles/s43856-025-00998-1) | 跨模态 |
| LDM 适配器视网膜增强 | [MedIA 2025](https://www.sciencedirect.com/science/article/abs/pii/S1361841525001501) | 潜在扩散分布适配 |
| 基线选项 | Real-ESRGAN / SwinIR | 通用 SR，报告 5.11.2 既定 |

综述两篇（Related Work 直接可引）：[Fundus IQA+IQE 综述 arXiv:2501.11520](https://arxiv.org/html/2501.11520v1)、[Springer 眼底增强综述 DOI 10.1007/s10791-025-09761-5](https://link.springer.com/article/10.1007/s10791-025-09761-5)。

## D. 数据侧

- **质量分桶训练/高质量子集微调**：作者观察"生成质量受训练数据质量影响大"的操作化——训练池按质量分桶，只用高质量桶再微调一轮，对照 D3。
- **A 域 replay 防遗忘**：D3/D4 均未做，微调线再启动时可加。
- 中文文献：[电子与信息学报·条件扩散小样本生成框架](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT240841?viewType=HTML)、[中国激光·扩散眼底融合](https://www.opticsjournal.net/Articles/OJe6590aea56454dad/Abstract)。
