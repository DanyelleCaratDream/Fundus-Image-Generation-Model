# 07 预训练生成模型调研（Phase D 前置）

> 回答"有没有适合的预训练生成模型，能直接或低成本地用在我们 severe 眼底图生成上"。
> 调研方式：**两遍**——先通用（非眼底），再眼底特化。优先 GitHub / HuggingFace / arXiv / 权威站点，**每个候选都核验过仓库在不在、权重下不下载得到、许可是什么**。
> 更新日期：2026-08-26
> 前置文档：`generate_project/pretrained/report-pretrained-lit.md`（Phase C3 预训练文献底稿，本报告补上它列的"SD+LoRA 8GB 可行性待调研"缺口）

---

## 0. 背景与硬约束（先读）

| 约束 | 数值/内容 | 影响 |
|:--|:--|:--|
| GPU | NVIDIA RTX 4060 Laptop **8GB**（~7GB 可用） | 过滤掉 12GB+ 的一切训练方案 |
| 输出分辨率 | **512×512 已锁定**（Phase D 决策） | 128 只是旧评估口径，预训练模型输出要对齐 512 |
| 生成目标 | severe 重度 DR 图（渗出/出血/微动脉瘤/棉絮斑） | 判别标准是病灶保留 + 抗识破 + 多样性 |
| 微调数据 | 330 张 web severe 图（低清）+ **老师数据集 train/severe 152 张**（可用，未定） | 数据量小 → 必须 LoRA/DreamBooth 这类低数据微调 |
| 现成条件 | 330 张血管骨架（ddpm conditions） | 血管条件生成是现成资产 |
| 评估 | 六维门控 + C2ST 必跑 + 复制检测 | 所有新方法都要过这套闸门 |

**前置结论（来自 `pretrained/report-pretrained-lit.md`，不重述）**：生成图增广的判别力不必然优于经典增广；价值锚定在"严重类平衡度/特异性"；评估必须患者级划分；眼底域特化预训练（RETFound-FD）确实拉开差距。

---

## 1. 第一遍：通用预训练生成模型

### 1.1 Stable Diffusion 1.5 / 2.1（LoRA 微调）—— 首选

- **是什么**：文本到图像的潜在扩散模型，是当前生态最成熟的开源生成器。SD 1.5 权重 `runwayml/stable-diffusion-v1-5`，SD 2.1-base `stabilityai/stable-diffusion-2-1-base`（512×512）。
- **VRAM（核验过）**：SD 1.5 **LoRA 训练 6GB 起、8GB 推荐**（Kohya_ss 官方口径）；RTX 3070 实测 ~7GB 能训。SD 2.1 与 1.5 同量级。推理 512×512 仅 4–6GB。
- **与我们**：LoRA 微调输出 512×512，正好对齐锁定的 Phase D 分辨率。8GB 卡可跑。DreamBooth（含 LoRA 变体）在 DR 上已有成功先例（见 2.2 的 ECC_DM_for_DR，66.84→74.20 平衡准确率）。
- **现成眼底 LoRA**（直接可用，见 2.1）：KylianSu `sd21_lora_fundus_v2`、AlexeyGHT ODiR LoRA。
- **结论**：**可行，且是本次调研的最优通用底座**。

### 1.2 Stable Diffusion XL（SDXL）—— 紧

- 推理 1024×1024 FP16 ~6.5GB（gigagpu 实测），**能跑但很紧**，需 xformers/SDP + `--medvram-sdxl`；训练/多 LoRA/带 refiner 会超 8GB。生态默认 1024 输出，反而要降采样到 512。
- **结论**：非必要不用。底座优先 1.5/2.1。

### 1.3 FLUX.1 —— 不可行

- 12B 扩散 Transformer。FP16 需 **24GB**；8GB 只能 NF4/GGUF Q4 **纯推理**（质量损失明显），LoRA 微调生态不如 SD。参考：Invoke 列 10GB 起，NIM 列 16GB 起。
- **结论**：直接排除。

### 1.4 ControlNet（结构条件）—— 与我们的血管骨架同构

- 给 SD 加"边缘/深度/骨架"等结构条件的官方方案（`lllyasviel/ControlNet`）。官方支持 **8GB**（`--medvram-sdxl`，lllyasviel 本人确认 6GB 低显存也能跑）。
- **与我们**：我们手里正好有 330 张血管骨架，血管条件生成（Canny/骨架 hint）是天然用法。眼底已有现成 VesselControlNet（2.4）。
- **注意**：ControlNet **训练**需要配对数据 + 更多显存；先用现成权重做推理最省。
- **结论**：可行，作后续强化项（路线 B）。

### 1.5 SD Inpainting / img2img（病灶插入）—— 与 poisson/retinex 同思路的升级版

- 用掩码指定区域，让模型补画内容。医学病灶插入已有多个公开先例：乳腺 MAM-E、甲状腺超声结节（DeepLabV3+ Dice 38→59）、病理 PathoGen。
- **与我们**：我们的 poisson 随机椭圆贴片、retinex 光照交换本质就是"病灶重排/插入"。SD inpaint 能生成**解剖上更合理的病灶**，且掩码可控（病灶位置/形状），8GB 推理可行，无需从零训练。
- **结论**：可行，作并行项（路线 C）。

### 1.6 StyleGAN2/3 —— 已知风险，低优先

- 512px 微调 VRAM 约 4–8GB 可接受，官方支持从 FFHQ 预训练 `--resume` 迁移 + ADA 抗小样本。
- **但我们已吃过亏**：本项目 DCGAN（12.9）/FiLM（37.2）等 GAN 变体全部落败，GAN 在 330 张眼底图上模式塌缩/过拟合是实测过的；且 GAN 没有条件生成病灶的结构优势。
- **结论**：不做 GAN 主路线，仅作反面对照。

### 1.7 其他通用模型—— 一句话排除

| 模型 | 为什么排除 |
|:--|:--|
| Kandinsky 2.2 | 与 SD 同级、生态和现成眼底权重都更少，且更吃显存 |
| DeepFloyd IF | 像素空间扩散，推理/微调都很重 |
| Würstchen / Stable Cascade | 官方 12GB+ |
| PixArt-α | 12GB+ |
| BigGAN | 只出 ImageNet 1000 类，无文本/结构条件 |

---

## 2. 第二遍：眼底特化预训练模型

### 2.1 现成眼底 LoRA / 扩散权重（下载即可用）

| 权重 | 底座 | 说明 | 许可 | 核验 |
|:--|:--|:--|:--|:--|
| [KylianSu/vessel-bezier-retinal-weights](https://huggingface.co/KylianSu/vessel-bezier-retinal-weights) | SD 2.1-base | `sd21_lora_fundus_v2`（rank64, noise_offset=0.1, ~51MB）+ G2/G3 血管 ControlNet + 下游分类器；NeurIPS 2026 提交配套 | **TBD（需联系作者）** | ✅ 权重在，restore 脚本在 |
| [AlexeyGHT/StableDiffusion_ODiR_lora-4](https://huggingface.co/AlexeyGHT) | SD | ODiR 视网膜数据集训练的 LoRA | — | ✅ 在 |
| [AlexeyGHT/Stable_Diffusion_v1.4_lora](https://huggingface.co/AlexeyGHT) | SD 1.4 | fundus LoRA | — | ✅ 在 |
| [AlexeyGHT/kandinsky2_2_*_ODiR](https://huggingface.co/AlexeyGHT) | Kandinsky 2.2 | ODiR 眼底 prior/decoder | — | ✅ 在 |

> ⚠️ KylianSu 是**论文复现套件**（要 SD2.1 base + EyePACS/APTOS/Messidor 原图 + 多条 restore 脚本），不是即插即用生成器；但它证明了"**SD2.1 + LoRA(rank32~64, noise_offset=0.1) + 血管 ControlNet**"在 DR 上整套可跑，且用了"a fundus photograph with diabetic retinopathy showing microaneurysms and hemorrhages"这类病灶文本。

### 2.2 眼底扩散（代码/论文在，权重需自己训或需高显存）

| 项目 | 方法 | 与我们的关系 | 可行性 |
|:--|:--|:--|:--|
| [CompDiff-fundus](https://huggingface.co/mahmoudibra98/compdiff-fundus) | SD 2.1-base 微调 + HCN 人口统计条件，512×512 | ⚠️ **青光眼域**（glaucoma/视杯视盘/近视），DR 病灶 prompt 是 out-of-distribution | 权重在，diffusers 可载；**领域不匹配，不直接用** |
| [ECC_DM_for_DR](https://github.com/AlanZhang1995/ECC_DM_for_DR)（MICCAI 2025） | 每 DR 级一个扩散模型（DreamBooth 式）+ 分类器语义筛选；DDR 上平衡准确率 66.84→74.20 | **思路最贴合我们**：按级生成 + 合成样本筛选 | **训练需 RTX 4090 24GB**（作者原话）；代码在、扩散权重未公开；**方法可借鉴，直接落地不行** |
| [OrdinalDiffusionModels](https://github.com/berenslab/OrdinalDiffusionModels)（berenslab） | 序数潜在扩散，DR 0–4 级序数条件；EyePACS 上 FID 降、QWK 0.79→0.87 | 序数条件比 one-hot 更贴合分级本质 | 源码在；需自训，显存未知 |
| [retinal_image_counterfactuals](https://github.com/berenslab/retinal_image_counterfactuals)（berenslab） | 扩散反事实生成（分类器引导） | 作者自述**严重/PDR 类受数据稀缺限制**——和我们的痛点一致，说明这类数据本来就难生成 | 源码在 |
| [DR-DDPM](https://github.com/Shashan-k/DR-DDPM) | 192px 纯 DDPM（APTOS），只有 notebook，无权重 | 我们已有更成熟的 FiLM DDPM | **无价值，跳过** |

### 2.3 眼底 GAN（论文/代码在，权重多不公开或需适配）

| 项目 | 方法 | 说明 |
|:--|:--|:--|
| [RetinaGAN](https://github.com/farrell236/RetinaGAN) | 条件 StyleGAN 生成 7 通道病灶图（按 ICDR 0–4 级）→ SPADE 转眼底图 | **唯一的"现成眼底 DR GAN"**：MIT 许可、checkpoints 在、Streamlit demo。2022 老栈（TF/PyTorch 混），需适配；GAN 通病风险同上 |
| FundusGAN（Ahn 2023, BSPC） | 半监督 GAN | 论文级，无易用权重 |
| CFIGGAN（2024） | mask 引导条件 GAN（血管树 + FOV + 4 类病灶） | 论文级 |
| DR-GAN | 双阶段，vessel+lesion mask 条件 + 分级向量 | 论文级 |
| Fundus GAN（IEEE 2022） | PUNet-33 血管分割 → 血管树转眼底；1:1 真实+合成混合训练最优 | 论文级 |
| Hou et al.（2023, BOE） | 条件 StyleGAN 病灶图 + GauGAN 转眼底 | 论文级，RetinaGAN 的完整版 |

> 2.3 共同点：思路全是对齐我们的（血管/病灶条件），但 **GAN 在我们 330 张上已被实测打爆**，且这些工作大多没给现成权重。参考价值 > 落地价值。

### 2.4 眼底 ControlNet（血管条件）

| 项目 | 方法 | 与我们的关系 |
|:--|:--|:--|
| VesselControlNet（KylianSu，[arXiv 2605.13015](https://arxiv.org/pdf/2605.13015)） | SD2.1 + 血管 Bézier hint ControlNet + LoRA(rank32, noise_offset=0.1)；channel-0 对血管曲折度不变 → 支持反事实扰动 | **与我们"血管骨架条件"完全同构**；证伪口径下血管几何对 DR 判别有因果贡献（Δ=+0.781） |
| [IEEE 2026 解剖保持 ControlNet](https://ieeexplore.ieee.org/document/11573738) | 血管边缘 ControlNet + 风格注入 + 掩码；青光眼筛查 AUC 0.7991→0.8525 | 血管拓扑保持的同时可控风格；论文级 |

> ⚠️ KylianSu 的 vessel LoRA 许可 TBD，商用/外发需先问作者。**但我们只做科研增广、数据不出仓库**，这个风险可控，仍需用户知晓。

### 2.5 分类器侧基础模型（Phase D 分类器用，非生成器）

| 模型 | 说明 | 用途 |
|:--|:--|:--|
| [RETFound](https://github.com/rmaphoh/RETFound_MAE)（[Nature 2023](https://www.nature.com/articles/s41586-023-06519-x)） | ViT-Large，160 万视网膜图 MAE 自监督预训练；CFP/OCT 权重（[HF open-eye/RETFound_MAE](https://huggingface.co/open-eye/RETFound_MAE)，**cc-by-nc-4.0**）；2025 更新版含 RETFound-DINOv2 系列 | Phase D 分类器骨干候选：**微调比 ResNet18 有更强的眼底先验**，与 ResNet18 基线同台对比；RETFound-FD 变体在眼底 RLAD 里 FID 30.3/79.7 优于 StyleGAN2 |

> 分类器微调加载很轻（ViT-Large 单卡可训），不影响生成侧预算。

---

## 3. 横向对比与推荐

### 3.1 三条候选路线

| 路线 | 做什么 | 8GB 可行性 | 与现有方法关系 | 阶段 |
|:--|:--|:--|:--|:--|
| **A. SD-LoRA 微调** | SD 2.1-base + LoRA 微调于 severe 眼底图（330 web + 老师 train severe），类 token 采样 512 | ✅ 训练 6–8GB、推理 4–6GB | 全新域（latent 扩散 vs 像素扩散 ddpm） | **主路线（先做）** |
| **B. 血管 ControlNet** | 现成 VesselControlNet / 自训血管 hint ControlNet + SD-LoRA | ✅ 推理 8GB；训练略紧 | 用上我们现成的 330 血管骨架 | 后续强化 |
| **C. SD 病灶插入** | SD inpaint/img2img 按掩码往真实眼底加病灶 | ✅ 推理 8GB | poisson/retinex 的"病灶重排"思路升级为真实感病灶 | 并行项 |

### 3.2 推荐：A 为主、C 并行、B 待定

1. **先做 A**：SD 2.1-base + LoRA（rank 64、noise_offset=0.1，照抄 KylianSu 已证实的配置）微调。这是唯一"下载即训、8GB 可跑、512 输出、有 DR 成功先例（ECC_DM 66.84→74.20）"的路线。
2. **C 并行**：做成 poisson/retinex 的平级竞争者进 TSTR/TRTR 对比，正好回答"病灶插入用 GAN/扩散比手工贴片强多少"。
3. **B 放 C3**：血管 ControlNet 等 A/C 跑完对比后再评估；许可证先跟 KylianSu 问。

### 3.3 与现有生成器的定位

| 生成器 | 性质 | 预期差异 |
|:--|:--|:--|
| poisson（80.8）/ retinex（72.5） | 真实病灶重排 | 真实感好但病灶组合受限（只"重排"不"新生成"） |
| FiLM DDPM（45.2） | 像素扩散，128px | 分辨率受限、病灶细节弱 |
| film/vae/dcgan 及 PCA/GMM/patch | 对照组 | 已知不可行 |
| **SD-LoRA（路线 A）** | latent 扩散，512px | 病灶是新合成的、分辨率够；风险是"领域漂移 + 伪影" |

---

## 4. 落地计划（草案，待用户确认）

> 遵循六段式汇报 + 每步确认 + C2ST 必跑 + 复制门控。

1. **数据决策（开放点，需用户拍板）**：LoRA 微调用哪些图？建议 **330 web severe + 老师 train/severe 152 张**（老师图分辨率 238-800px，需预处理到 512）；仅用 330 张低清 web 图可能不够。
2. 环境：SD 2.1-base 权重 + diffusers + 8GB 实测显存。
3. 微调：LoRA rank 64、noise_offset=0.1、类 token（如 "severe DR fundus"）；1024 步级起步。
4. 采样 512×512 → 与评估口径对接（下采样 128 过现有六维门控 + C2ST 必跑 + 复制检测三件套）。
5. 进 Phase D TSTR/TRTR 对比（与 poisson/retinex/film 同台，分类器 = ResNet18 基线 + RETFound 可选骨干）。
6. 门控：复制率超标 / 意外高分 → 停下汇报。

---

## 5. 风险与对策

| 风险 | 说明 | 对策 |
|:--|:--|:--|
| 领域漂移 / 伪影 | 8GB + 330 张微调 SD，容易学到"伪眼底" | 补老师真实 severe 数据；质量筛选（C2ST、病灶指标、分类器预测筛选——照 ECC 的 semantic filtering）；人眼评分【待人工评估】 |
| 记忆复制 | 小样本微调易背图 | 现有 NN-SSIM>0.85 近复制过滤 + max_retry 重采样可直接复用 |
| 许可 | SD OpenRAIL；KylianSu TBD；RETFound cc-by-nc-4.0；老师数据是学校数据 | **全程本机训练、数据不出仓库**；外发/商用前确认（现阶段纯科研无碍） |
| 文本条件弱 | DR 病灶文本提示不精确 | 类 token + 固定模板（"a fundus photograph with diabetic retinopathy showing microaneurysms and hemorrhages"） |
| 增广不必然胜出 | pretrained lit 的既有警告 | 锚定严重类平衡度/特异性，不吹全面超越；严格患者级划分 |

---

## 6. 来源链接

**通用层**
- SD 1.5 LoRA 8GB：[Kohya_ss 8GB 实报（知乎）](https://zhuanlan.zhihu.com/p/609632788)、[SD VRAM 指南（SynpixCloud）](https://www.synpixcloud.com/blog/stable-diffusion-gpu-requirements-guide)
- SDXL 8GB：[RTX 4060 跑 SDXL（gigagpu）](https://gigagpu.com/can-rtx-4060-run-stable-diffusion-xl/)、[SDXL VRAM（gigagpu）](https://gigagpu.com/sdxl-vram-requirements/)
- FLUX：[FLUX VRAM（evezone）](https://evezone.evetech.co.za/daily-drop/new-flux-image-models-and-the-vram-they-demand-on-local-rigs)、[Invoke 硬件要求](http://invoke.ai/start-here/system-requirements/)
- ControlNet 8GB：[lllyasviel 官方 8GB 确认（GitHub discussion #2086）](https://github.com/Mikubill/sd-webui-controlnet/discussions/2086)、[ControlNet 官方仓库](https://github.com/lllyasviel/ControlNet)
- SD 病灶插入先例：[PathoGen（病理）](https://huggingface.co/mkoohim/PathoGen)、[甲状腺超声（ScienceDirect）](https://www.sciencedirect.com/science/article/abs/pii/S1532046425001923)
- StyleGAN：[官方仓库](https://github.com/NVlabs/stylegan3)、[StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada)

**眼底层**
- KylianSu vessel LoRA + ControlNet：[HF 权重](https://huggingface.co/KylianSu/vessel-bezier-retinal-weights)、[代码 reproduce_dr](https://github.com/KylianSu/reproduce_dr)、[arXiv 2605.13015](https://arxiv.org/pdf/2605.13015)
- AlexeyGHT 眼底 LoRA：[HF 用户页](https://huggingface.co/AlexeyGHT)
- CompDiff-fundus：[HF 模型](https://huggingface.co/mahmoudibra98/compdiff-fundus)
- ECC_DM_for_DR：[GitHub](https://github.com/AlanZhang1995/ECC_DM_for_DR)、[MICCAI 2025 论文页](https://papers.miccai.org/miccai-2025/0146-Paper4449.html)
- OrdinalDiffusion：[GitHub](https://github.com/berenslab/OrdinalDiffusionModels)、[arXiv 2602.24013](https://arxiv.org/html/2602.24013v2)
- 眼底反事实：[GitHub（berenslab）](https://github.com/berenslab/retinal_image_counterfactuals)
- RetinaGAN：[GitHub](https://github.com/farrell236/RetinaGAN)
- 眼底 ControlNet（IEEE 2026）：[IEEE 11573738](https://ieeexplore.ieee.org/document/11573738)
- RETFound：[GitHub](https://github.com/rmaphoh/RETFound_MAE)、[HF](https://huggingface.co/open-eye/RETFound_MAE)、[Nature 2023](https://www.nature.com/articles/s41586-023-06519-x)
- DR-DDPM：[GitHub](https://github.com/Shashan-k/DR-DDPM)

**前置证据链**
- `generate_project/pretrained/report-pretrained-lit.md`（Phase C3 文献底稿，本报告补其"SD+LoRA 8GB 可行性待调研"缺口）
