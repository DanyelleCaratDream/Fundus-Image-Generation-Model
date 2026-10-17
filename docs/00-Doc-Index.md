# 工程文档导航 + 文件路径索引

> 这是项目的「文档地图」。新会话（或新成员）先读本文件定位所有关键文档与路径。
> 更新日期：2026-10-17（v3.1：登记论文草稿区 `paper/`；v3：补 11-15 号文档 + 路径全面核对；2026-10-16 文档体系已按**论文第一**重整，见 [14-Paper-Spine](14-Paper-Spine.md)）

---

## 一、项目一句话

**FundusGen** — 小样本眼底彩照生成研究：失效机理（failure mechanisms）与医学语义评估（evaluation），
并以可控 severe 图生成完成 DR 分级数据扩充与下游验证（TSTR）。**论文第一、实用性第二**（2026-10-16 定位）。

---

## 二、文档地图

### docs/（规范文档中心）

| 文档 | 内容 | 状态 |
|------|------|:--:|
| [00-Doc-Index.md](00-Doc-Index.md) | 本文档：导航 + 路径索引 | ✅ 2026-10-16 |
| [01-Project-Overview.md](01-Project-Overview.md) | 项目需求与目标（v7：论文第一 + Phase D 实况） | ✅ 2026-10-16 |
| [02-Technical-Stack.md](02-Technical-Stack.md) | 技术栈、依赖、显存预算 | ✅ |
| ~~03-Development-Standards.md~~ | 已归档 → `_archive/docs/`（内容被 REPORT + Work-Guide 覆盖，路径失效） | 📦 2026-10-16 |
| ~~04-Training-Pipeline.md~~ | 已归档 → `_archive/docs/`（同上） | 📦 2026-10-16 |
| ~~05-Research-Methodology.md~~ | 已归档 → `_archive/docs/`（被 evaluation_report.md 覆盖） | 📦 2026-10-16 |
| [06-Execution-Plan.md](06-Execution-Plan.md) | 执行路线图（v3 实况：A/B/C 完成，D 进行中 + 论文线） | ✅ 2026-10-16 |
| [07-Model-Card-Template.md](07-Model-Card-Template.md) | 模型档案模板 | ✅ |
| [08-Work-Guide.md](08-Work-Guide.md) | 工作说明手册（常用命令 + 已知坑） | ✅ |
| [09-Score-Scheme-Design.md](09-Score-Scheme-Design.md) | 综合评分方案设计（决策链/权重理由/门控） | ✅ 2026-08-02 |
| [10-Score-Scheme-For-Beginners.md](10-Score-Scheme-For-Beginners.md) | 评分标准白话版（故事+类比） | ✅ 2026-08-02 |
| [11-Paper-Readiness.md](11-Paper-Readiness.md) | 论文就绪度盘点（证据家底 / 缺口 / 报告诊断 / 路线 A+B） | ✅ 2026-10-15 |
| [12-Literature-Survey-and-Positioning.md](12-Literature-Survey-and-Positioning.md) | 范文调研与论文定位（30+ 篇：哪些贡献已有前人、真正的空白） | ✅ 2026-10-15 |
| [13-Expert-Evaluation-Protocol.md](13-Expert-Evaluation-Protocol.md) | 眼科医生盲评方案（草案，待导师/医生确认） | 🟡 草案 |
| **[14-Paper-Spine.md](14-Paper-Spine.md)** | ⭐ **论文脊柱**：主张→证据映射 + 口径统一表 + 术语中英对照——写作宪法 | ✅ 2026-10-16 |
| **[15-Data-Documentation.md](15-Data-Documentation.md)** | ⭐ **数据权威文档**：全部数据池 + 卫生结论 + 泄漏清单 + 转告老师事项 | ✅ 2026-10-16 |

### 实验与状态（数字唯一来源）

| 文件 | 说明 |
|------|------|
| `generate_project/deep_learning/EXPERIMENTS.md` | ⭐ 深度学习实验登记表（DL-EX-001~020），**所有数字从这里取** |
| `generate_project/machine_learning/EXPERIMENTS.md` | 传统 ML 实验登记表（EX-001~005；原 `docs/06-Records.md` 升格） |
| `research-report/CONTINUE_GUIDE_NEW.md` | 会话续接指引（v24：只记位置+下一步+纪律，不复制数字） |

### research-report/（交付物）

| 文件 | 说明 |
|------|------|
| `REPORT_DL.docx` | 深度学习生成研究报告（VAE / GAN / Diffusion 全记录；实验日志体裁，论文按 docs/14 重组） |
| `REPORT_ML.docx` | 传统 ML 生成研究报告（Phase C2） |
| `evaluation_report.md` / `.docx` | 评估报告 + 评分标准调研合并版（Phase A 结果 + Phase B 结论） |
| `interview-prep.docx` | 面试/答辩追问准备（Q1-Q14） |
| `figures/` | 报告嵌入结果图（`eval/plot_metrics.py` + `score_scheme.py --scorecard` 生成） |
| `REPORT_DL_text.txt` | REPORT_DL.docx 纯文本提取版（2026-10-17 入库备份，论文写作引用源；**停更于 2026-08-02**，D1–D4 与数据卫生不在此文件，须引 EXPERIMENTS.md/docs/15） |

### paper/（论文正文草稿区，2026-10-17 新建）

| 文件 | 说明 |
|------|------|
| `paper/README.md` | 写作约定（数字三点头、分数标尺度、占位语法、引用红线） |
| `paper/outline.md` | ⭐ 全书节级大纲（单一真源）+ 占位清单 |
| `paper/abstract.md` + `paper/ch1…ch7` | 各章草稿（首批：失效机理链章节组 3.3/3.4/4.1/4.3/5.1–5.6/6.1/6.2） |
| `paper/refs-notes.md` | 引用便签（.bib 建库前暂存，R1–R6） |
| `paper/figures/` | 图规格约定（≥300 DPI、(a)(b)(c) 标号）+ 占位清单 |

### 其他入口

| 位置 | 说明 |
|------|------|
| `classify_project/README.md` | Phase D 方案（替换 76 张翻转副本 + TSTR 协议 + 硬约束） |
| `literature/` | 文献管理（links.md + pdf/） |
| `generate_project/pretrained/report-pretrained-lit.md` 等 | 预训练/迁移文献归档；SD-LoRA 调研 = `machine_learning/report/07-pretrained-gen-models.md` |
| `_archive/ARCHIVE.md` | 归档登记（10-15 瘦身 + 10-16 两批文档归档） |

> **📁 各方法目录的文档约定（2026-10-16 定）**：每个方法目录只保留三件套——`README.md`（入口/状态）＋ `EXPERIMENTS.md`（实验登记，如适用）＋ `report/`（文献与报告）。
> **不再建方法级 `docs/` 子目录**（工程规范统一收敛到根 `docs/`；`generate_project/docs/` 宏观 4 份与 `machine_learning/docs/` 00-05 已归档）。

---

## 三、关键路径索引

> 完整数据池描述（含卫生状态）→ **[15-Data-Documentation.md](15-Data-Documentation.md)**

### 数据集（git 排除）
| 路径 | 内容 |
|------|------|
| `fundus/_all_images_ORIGINAL/` | A 域网络图（方形 JPG）——**现存 299 张**（评估快照 `eval_data/real` 330 张，差异待确认） |
| `fundus/all_ORIGINAL_sq256/` | 163 张手挑池（153 唯一） |
| `fundus/combined_sq256/` | **193 张**合并训练池（D1/D2 与正式生成用）+ `conditions_combined/` 193 mask |
| `fundus/HRSG_4_758_v0.3.4/` | 老师官方数据集 758 张（train 552/val 135/test 71）——**只读** |
| `fundus/severe_train_sq256/` `sq128/` | train severe 唯一底图 76 张（派生） |
| `fundus/_removed_dups_sq256/` | 墓园（清理移出文件 + MANIFEST.md，可回滚） |
| ~~`fundus/_all_images_raw/_256/_much`~~ | 已移出磁盘（2026-10 瘦身）——旧文档引用以此为准 |

### 条件图 / 骨架
| 路径 | 内容 |
|------|------|
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/` | 330 张血管骨架 mask（A 域条件扩散 + Vessel Dice 用） |
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/prep_conditions.py` | 血管骨架提取脚本 |

### 模型 checkpoint（可评估的 6 个）
| 模型 | 人工分 | 六维分（11 模型现版） | checkpoint 路径 |
|------|:--:|:--:|------|
| FiLM+L1+LPIPS（最佳深度） | 87.5 | 45.2 | `ddpm/results_film_l1lpips/models/final_model.pth` |
| FiLM DDPM MSE | 85 | 37.2 | `ddpm/results_film/models/final_model.pth` |
| 条件扩散 | 70 | 35.0 | `ddpm/results_cond/models/final_model.pth` |
| 基础 DDPM（去 CJ） | 75 | 34.5 | `ddpm/results_去掉ColorJitter版/models/final_model.pth` |
| DCGAN | 20 | 12.9 | `Fundus-GAN/dcgan/results_220726_020708/models/checkpoint_epoch_001500.pth` |
| VAE Large 800 | 10 | 26.6 | `Fundus-VAE/results/vanilla_vae_large_210726_212102/models/final_model.pth` |

> ⚠️ 两套分数定义见 [14-Paper-Spine.md](14-Paper-Spine.md) 口径表 #2，不可混排。
> WGAN-GP / StyleGAN2 / VAE 1200 无 checkpoint（用户确认跳过）。

### 评估
| 路径 | 内容 |
|------|------|
| `eval_data/` | real 快照（330）+ 各模型生成图（JSON 已提交，图片 gitignore）+ `_scores.json` |
| `eval/metrics_common.py` / `metrics_fundus.py` | 两层指标脚本，用法见 `eval/README.md` |
| `eval/score_scheme.py` | 综合评分（六维门控），设计动机 docs/09 |
| `eval/plot_metrics.py` + `score_scheme.py --scorecard` | 可视化（自动扫 JSON，新方法重跑即入图） |
| `eval/human_scores.csv` | ⭐ 人工分唯一来源（2026-10-16 起替代硬编码） |

### 环境
| 路径 | 内容 |
|------|------|
| `C:\Users\Yelle\.claude\projects\d--AI-Model-Project-for-Fundus-Color-Images\memory\` | Claude 持久记忆（MEMORY.md 索引） |

---

## 四、更新约定

- **数字**：只在两份 EXPERIMENTS.md / eval_data JSON / human_scores.csv 维护，其他文档引用不复制
- **每轮工作结束**：更新 `research-report/CONTINUE_GUIDE_NEW.md`（版本号 + 日期 + 当前位置）
- **doc 增删**：同步更新本文件「文档地图」；**路径变动**：同步「关键路径索引」
- **口径**：拿不准的口径查 [14-Paper-Spine.md](14-Paper-Spine.md) 第三节；新口径先入表再使用
