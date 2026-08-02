# 工程文档导航 + 文件路径索引

> 这是项目的「文档地图」。新会话（或新成员）先读本文件定位所有关键文档与路径。
> 更新日期：2026-08-01

---

## 一、项目一句话

**FundusGen** — 用 330 张严重症状眼底彩照，训练生成模型产出「相似但不相同」的重度眼底图，用于**扩充数据集、训练 DR 分级分类器**（可能也做病灶类型判断）。

---

## 二、文档地图

### docs/（工程规范中心）

| 文档 | 内容 | 状态 |
|------|------|:--:|
| [00-Doc-Index.md](00-Doc-Index.md) | 本文档：导航 + 路径索引 | ✅ |
| [01-Project-Overview.md](01-Project-Overview.md) | 项目需求与目标（新方向 + 老师反馈） | ✅ |
| [02-Technical-Stack.md](02-Technical-Stack.md) | 技术栈、依赖、显存预算 | ✅ |
| [03-Development-Standards.md](03-Development-Standards.md) | 工程规范：目录/CLI/代码风格 | ✅ |
| [04-Training-Pipeline.md](04-Training-Pipeline.md) | 各模型训练管线设计 | ✅ |
| [05-Research-Methodology.md](05-Research-Methodology.md) | 评估体系（两层指标：通用 + 专用） | ✅ |
| [06-Execution-Plan.md](06-Execution-Plan.md) | 执行路线图（Phase A/B/C/D + 状态） | ✅ |
| [07-Model-Card-Template.md](07-Model-Card-Template.md) | 模型档案模板 | ✅ |
| [08-Work-Guide.md](08-Work-Guide.md) | 工作说明手册（常用命令 + 已知坑） | ✅ |
| [09-Score-Scheme-Design.md](09-Score-Scheme-Design.md) | 综合评分方案设计文档（为什么这么设计：决策链/权重理由/门控/归一化/指标去留/局限）（Word 版：09-Score-Scheme-Design.docx） | ✅ 2026-08-02 |
| [10-Score-Scheme-For-Beginners.md](10-Score-Scheme-For-Beginners.md) | 评分标准白话版（用故事+类比解释评分为什么这么设计，面向看不懂正式报告的人） | ✅ 2026-08-02 |

### research-report/（交付物 + 会话指引）

| 文件 | 说明 |
|------|------|
| `CONTINUE_GUIDE_NEW.md` | ⭐ 会话续接指引（当前状态 + 下一步，compact 后先读它） |
| `REPORT（原版）.docx` | 科研报告（用户精修终版） |
| `眼底彩照生成模型.pptx` | 29 页答辩 PPT（用户排版中） |
| `interview-prep.md` / `.docx` | 面试追问准备文档（Q1-Q14） |
| `speaking_script.md` | 答辩讲稿大纲 |
| `evaluation_report.md` / `.docx` | 评估报告 + 评分标准调研合并版（Phase A 结果 + Phase B 结论） |

---

## 三、关键路径索引

### 数据集
| 路径 | 内容 |
|------|------|
| `fundus/_all_images_ORIGINAL/` | **330 张严重症状眼底图**（方形 JPG，训练基准集） |
| `fundus/_all_images_raw/` | 330 张原始采集图（未裁剪，分辨率各异） |
| `fundus/_all_images_256/` | 1320 张 256×256（每张 4 几何变体） |
| `fundus/_all_images_much/` | 1320 张原分辨率增强版 |

### 条件图 / 骨架
| 路径 | 内容 |
|------|------|
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/` | 330 张 128×128 血管骨架 mask（条件扩散用 + 评估 Vessel Dice 用） |
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/prep_conditions.py` | 血管骨架提取脚本（绿通道+CLAHE+Top-hat+自适应阈值+骨架化） |

### 模型结果与 checkpoint（可评估的 6 个）
| 模型 | 评分 | checkpoint 路径 | 架构要点 |
|------|:--:|------|------|
| FiLM+L1+LPIPS（最佳） | 85~90 | `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_film_l1lpips/models/final_model.pth` | base_dim=128, FiLM, 条件 |
| FiLM DDPM MSE | 85 | `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_film/models/final_model.pth` | base_dim=128, FiLM, 条件 |
| 条件扩散 | 70 | `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_cond/models/final_model.pth` | base_dim=64, 条件 |
| 基础 DDPM（去 CJ） | 75 | `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_去掉ColorJitter版/models/final_model.pth` | base_dim=64, 无条件 |
| DCGAN | 20 | `generate_project/deep_learning/Fundus-GAN/dcgan/results_220726_020708/models/checkpoint_epoch_001500.pth` | latent_dim=100 |
| VAE Large 800 | 10 | `generate_project/deep_learning/Fundus-VAE/results/vanilla_vae_large_210726_212102/models/final_model.pth` | latent_dim=256, dim=64 |

> ⚠️ **无 checkpoint 无法评估**：WGAN-GP、StyleGAN2、VAE Large 1200（权重已丢失，用户确认跳过）

### 评估
| 路径 | 内容 |
|------|------|
| `eval_data/` | 评估结果（real/ + 每模型 300 张生成图已产出；图片 .gitignore 只提交 `*_metrics.json` + `_scores.json`） |
| `eval/metrics_common.py` / `metrics_fundus.py` | 两层指标脚本（通用层 + 专用层），用法见 `eval/README.md` |
| `eval/score_scheme.py` | 综合评分（人工分校准 + 六维门控 0-100 总分），设计动机见 `docs/09`，报告 5.6/5.7/5.8 |

### 环境
| 路径 | 内容 |
|------|------|
| `C:\Users\Yelle\.claude\projects\d--AI-Model-Project-for-Fundus-Color-Images\memory\` | Claude 持久记忆（MEMORY.md 索引） |

---

## 四、更新约定

- **每次任务结束**：更新 `research-report/CONTINUE_GUIDE_NEW.md`（当前状态 + 下一步）
- **doc 增删**：同步更新本文件的「文档地图」
- **路径变动**：同步更新「关键路径索引」
