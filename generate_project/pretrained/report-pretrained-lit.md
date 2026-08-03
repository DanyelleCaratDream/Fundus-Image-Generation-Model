# pretrained/ 预训练与迁移学习：文献资料归档

> 用途：给 Phase C3（Stable Diffusion + LoRA 等预训练方法）留的资料底稿。
> 来源：Phase B 调研（evaluation_report.md 第二部分）+ 2026-08-03 文献调研（小样本医学增广/生成模型扩充证据）+ 02 篇眼底新奇组合。
> 更新日期：2026-08-03

---

## 0. 一句话现状

**预训练方向是"可探索项"而非"必选项"**：Phase B 结论是医学预训练不必然更好（Woodland 反直觉证据），但眼底域特化预训练（RETFound-FD）确实能拉开差距。启动条件：Phase C2（传统 ML）完成后按需评估。

---

## 1. 预训练在生成/判别上的已知结论（Phase B 已落地的）

1. **域适配特征提取器有争议，不必然更好**——Woodland et al. (MICCAI 2024)：4 医学模态 × 11 提取器对比专家目视，**ImageNet 预训练提取器排名稳定且与专家一致（SwAV 显著相关），RadImageNet（医学预训练）反而波动大、与专家相矛盾**。
2. **眼底域特化预训练确实能拉开差距**：RETFound-FD（160 万视网膜图预训练）在眼底域优势明显（RLAD: FID 30.3/RET-FD 79.7 vs StyleGAN2 98.1/116.0）→ 域适配是"可探索项"。
3. **明确不做**：用域适配提取器重算全部历史模型（Woodland 证据 + 工作量大）。

> 详见 `research-report/evaluation_report.md` 第二部分 + `docs/05-Research-Methodology.md`。

---

## 2. 生成模型扩充（GAN/扩散）与预训练骨干：文献证据（2026-08-03 调研补）

**核心结论**：把小样本医学图像上的生成模型扩充与经典增广直接对比，**生成图扩充的判别力提升普遍不优于经典增广**；其价值集中在改善类不平衡/特异性/校准类指标，且仅在"与经典增广组合 + 样本筛选"时最有效。若坚持用生成模型扩充，**扩散（DDPM）优于 GAN**。

| 文献 | 发现 |
|:--|:--|
| **Frid-Adar et al.（2018）**，Neurocomputing 321:321–331，DOI [10.1016/j.neucom.2018.09.013](https://doi.org/10.1016/j.neucom.2018.09.013) [V] | 经典正例：182 肝脏病灶，经典增广（78.6/88.4 灵敏/特异）→ 加 GAN 图后 85.7/92.4 |
| **Mantegna et al.（2024）**，ICPR 2024 Workshops，DOI [10.1007/978-3-031-87660-8_7](https://doi.org/10.1007/978-3-031-87660-8_7) [V] | **696 实验、3 种 GAN、6 个 MedMNIST 集**：GAN 增广低维尚可，经典增广仍整体胜出 |
| **Fernández Santana et al.（2026）**，Knowledge-Based Systems（[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705126013663)）[V] | StyleGAN2-ADA 图 50–200%，AUC 几乎不动（~0.70），但特异性/平衡准确率/MCC 提升；**图像级评估高估 8–12pp → 必须患者级划分** |
| **Noriega Cedeño et al.（2026）**，arXiv [2605.23094](https://arxiv.org/abs/2605.23094) [V] | 架构与比例依赖；**视觉保真（FID/KID）不预测下游有用性**；合成图仍可被机器区分（57.7%>随机） |
| **（2024）**arXiv [2412.12532](https://arxiv.org/abs/2412.12532) [V] | 生成族内部：DDPM 比 PGGAN 更真实且分类提升更大（+6%） |
| **Babu et al.（2024）**，arXiv [2407.21674](https://arxiv.org/abs/2407.21674) [V]，DEMI@MICCAI 2024 | **simplicity bias**：模型利用"真实 vs 合成"统计差异，源与标签相关时训练好、部署崩 |

**对本项目的含义**：
- 现有 6 个深度生成方法（VAE/DCGAN/DDPM）不追求"超越经典增广"，而应作为**经典增广的叠加项 + 质量筛选**。
- 若后续做预训练生成（Phase C3 SD LoRA），预期价值锚定在"改善严重类特异性/平衡度"而非"AUC 全面超越"。
- 评估必须**患者级划分**（本项目 330 张若含同一患者双眼底，图像级评估会系统性高估）。

---

## 3. 眼底"结构先行"深度范式的预训练启示（02 篇补）

深度方法证明"结构图先行 + 上色/转图"在眼底是主流且有效，这为非深度组合获取结构 mask 提供现成答案（直接跑分割器）：

| 文献 | 相关点 |
|:--|:--|
| **Costa et al. 2018**，IEEE TMI 37(3):781-791，DOI [10.1109/TMI.2017.2759102](https://pubmed.ncbi.nlm.nih.gov/28981409/) [V] | 对抗自编码器生成血管树 + GAN 上色；生成图与训练集"substantially different"仍解剖一致 |
| **Alimanov & Islam 2023**，ICCP，DOI [10.1109/ICCP56744.2023.10233841](https://ar5iv.labs.arxiv.org/html/2308.08339) [V] | 两阶段 DDPM：先生成血管树 → 条件转眼底图 |
| **Fhima et al. 2025**，arXiv [2503.01190](https://ar5iv.labs.arxiv.org/html/2503.01190) [V] | RLAD：潜在扩散 + 条件化血管/视盘视杯/病变多布局，下游分割 +8.1% |
| **Brown et al. 2024**，Nat Commun 15:6859，DOI [10.1038/s41467-024-50911-y](https://preview-www.nature.com/articles/s41467-024-50911-y) [V] | PI-GAN：生物物理规则生成血管树 + CycleGAN 转眼底风格；仅 100 张仿真图训练即近 SOTA |

---

## 4. 待补资料（后续方向）

- **Stable Diffusion + LoRA 在 8GB VRAM 的可行性**（Phase C3 启动前调研：LoRA 512 输出 vs 128 评估体系对接）——尚未调研，列入 C3 前任务。
- 眼底域特化预训练基础模型（RETFound 系列）的最新进展。
