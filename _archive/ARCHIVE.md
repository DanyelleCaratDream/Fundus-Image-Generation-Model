# 归档记录

> 归档日期：**2026-10-15** ｜ 执行人：Claude（经用户逐项确认）

为「正式生成图片 + 写论文」做准备，对项目做了一次瘦身。**归档而非删除**，全部可从下表的原路径恢复。

仓库体积：**38,684 MB → 约 6,130 MB**

---

## 一、大文件（已移出仓库）

**位置：`D:/FundusGen_archive/`** —— 移出仓库是为了真正释放磁盘，且它们本来就被 `.gitignore` 排除，不影响 git 历史。

### 1.1 冗余模型权重（31.22 GB，49 个 `.pth`）

每个实验只保留 `final_model.pth`（评估所用的那个），其余训练中间检查点全部归档。

| 实验目录 | 归档数量 | 大小 | 归档的文件 |
|---|---:|---:|---|
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_cond` | 8 | 2.48 GB | `checkpoint_INTERRUPTED.pth`, `checkpoint_epoch_000150.pth`, `checkpoint_epoch_000300.pth`, `checkpoint_epoch_000450.pth`, `checkpoint_epoch_000500.pth`, `checkpoint_epoch_000600.pth`, `checkpoint_epoch_000700.pth`, `checkpoint_epoch_000800.pth` |
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_film` | 6 | 7.76 GB | `checkpoint_epoch_000200.pth`, `checkpoint_INTERRUPTED.pth`, `checkpoint_epoch_000250.pth`, `checkpoint_epoch_000300.pth`, `checkpoint_epoch_000400.pth`, `checkpoint_epoch_000500.pth` |
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_film_l1lpips` | 13 | 16.82 GB | `checkpoint_epoch_000060.pth`, `checkpoint_INTERRUPTED.pth`, `checkpoint_epoch_000120.pth`, `checkpoint_epoch_000180.pth`, `checkpoint_epoch_000240.pth`, `checkpoint_epoch_000300.pth`, `checkpoint_epoch_000360.pth`, `checkpoint_epoch_000420.pth`, `checkpoint_epoch_000480.pth`, `checkpoint_epoch_000540.pth`, `checkpoint_epoch_000600.pth`, `checkpoint_epoch_000660.pth`, `checkpoint_epoch_000780.pth` |
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_去掉ColorJitter版` | 8 | 2.48 GB | `checkpoint_epoch_000150.pth`, `checkpoint_epoch_000300.pth`, `checkpoint_epoch_000450.pth`, `checkpoint_epoch_000600.pth`, `checkpoint_epoch_000750.pth`, `checkpoint_epoch_000900.pth`, `checkpoint_epoch_001050.pth`, `checkpoint_epoch_001200.pth` |
| `generate_project/deep_learning/Fundus-GAN/dcgan/results_210726_221429` | 4 | 0.51 GB | `checkpoint_INTERRUPTED.pth`, `checkpoint_epoch_000200.pth`, `checkpoint_epoch_000400.pth`, `checkpoint_epoch_000600.pth` |
| `generate_project/deep_learning/Fundus-GAN/dcgan/results_210726_233005` | 1 | 0.05 GB | `checkpoint_INTERRUPTED.pth` |
| `generate_project/deep_learning/Fundus-GAN/dcgan/results_220726_020708` | 3 | 0.35 GB | `checkpoint_INTERRUPTED.pth`, `checkpoint_epoch_000500.pth`, `checkpoint_epoch_001000.pth` |
| `generate_project/deep_learning/Fundus-VAE/results/test` | 2 | 0.04 GB | `final_model.pth`, `checkpoint_epoch_000005.pth` |
| `generate_project/deep_learning/Fundus-VAE/results/vanilla_vae_large_210726_212102` | 4 | 0.72 GB | `checkpoint_epoch_000250.pth`, `checkpoint_epoch_000500.pth`, `checkpoint_epoch_000750.pth`, `checkpoint_epoch_000800.pth` |

**合计 49 个文件，31.22 GB**

**恢复方式**：从 `D:/FundusGen_archive/weights/<原相对路径>` 拷回仓库同路径即可。

**保留的 7 个权重**（未动）：

| 模型 | 权重 |
|---|---|
| FiLM+L1+LPIPS（最佳，85–90 分） | `ddpm/results_film_l1lpips/models/final_model.pth` |
| 　└ 另留最佳 epoch | `ddpm/results_film_l1lpips/models/checkpoint_epoch_000720.pth` |
| FiLM DDPM MSE（85 分） | `ddpm/results_film/models/final_model.pth` |
| 条件扩散（70 分） | `ddpm/results_cond/models/final_model.pth` |
| 基础 DDPM 去 CJ（75 分） | `ddpm/results_去掉ColorJitter版/models/final_model.pth` |
| DCGAN（20 分） | `Fundus-GAN/dcgan/results_220726_020708/models/checkpoint_epoch_001500.pth` |
| VAE Large 800（10 分） | `Fundus-VAE/results/vanilla_vae_large_210726_212102/models/final_model.pth` |

> 多留 `checkpoint_epoch_000720.pth` 的原因：`REPORT_DL.docx` §5.10.2 称「图 13a 是 epoch 700（最佳效果）」，但检查点按 60 的倍数保存（000060…000780），**没有 700**。为免论文复现那张图时找不到对应权重，两个都留。

### 1.2 重复数据（551 MB）

| 文件 | 原因 |
|---|---|
| `fundus/HRSG_4_758_v0.3.4.rar` | 与已解压的同名目录 `fundus/HRSG_4_758_v0.3.4/` 内容重复，解压版才是在用的 |

**位置**：`D:/FundusGen_archive/data/HRSG_4_758_v0.3.4.rar`

---

## 二、小文件（已移入本目录）

| 归档内容 | 原路径 | 归档理由 |
|---|---|---|
| `ppt/暑期总结_深度学习.pptx` | `research-report/暑期总结_深度学习.pptx` | 用户确认暑期汇报 PPT 已不需要 |
| `ppt/_build_summer_ppt.py` | `research-report/_build_summer_ppt.py` | 上述 PPT 的生成脚本 |
| `ppt/_prep_ppt_assets.py` | `research-report/_prep_ppt_assets.py` | 上述 PPT 的素材预处理脚本 |
| `docs/CONTINUE.md.obsolete` | `.claude/CONTINUE.md` | 文件开头自我标注「已过时」，被 `CONTINUE_GUIDE_NEW.md` 取代 |
| `docs/09-Score-Scheme-Design.docx` | `docs/09-Score-Scheme-Design.docx` | 与同名 `.md` 内容重复，只保留 md |
| `scripts/rotate_augment_check.py` | `fundus/_all_images_ORIGINAL/rotate_augment_check.py` | 交互式人工检查工具，非流程必需 |
| `scripts/make_comparison.py` | `generate_project/machine_learning/scripts/make_comparison.py` | 一次性出图脚本 |
| `scripts/make_result_figs.py` | `generate_project/machine_learning/scripts/make_result_figs.py` | 一次性出图脚本 |

**恢复方式**：`mv _archive/<子目录>/<文件> <原路径>`

---

## 三、直接删除（未归档）

| 内容 | 理由 |
|---|---|
| `research-report/ppt_assets/`（26.8 MB） | PPT 素材，可由 `_prep_ppt_assets.py` 从源数据重新生成 |
| `research-report/__pycache__/` | Python 编译缓存 |
| `Fundus-VAE/{beta_vae,utils,vanilla_vae}/`、`ddpm/eval_data/` | 4 个**空目录**（0 条目），从未有内容 |

---

## 四、明确保留未动

- **全部代码**：各家族 `train.py` / `generate.py` / `prep_conditions.py` / ML 的 `*_gen.py`

- **数据**：`fundus/_all_images_ORIGINAL/`（330 张）、`ddpm/conditions/`（330 张血管骨架）

- **实验记录**：各 `results_*/` 的 `config.json`、`logs/`、**`images/`**（论文里图 1–13 的原始素材）

- **评估**：`eval/` 四个脚本、`eval_data/`（24 个 JSON + 样本图）

- **报告**：`REPORT_DL.docx`、`REPORT_ML.docx`、`evaluation_report.md/.docx`、`CONTINUE_GUIDE_NEW.md`、`interview-prep.docx`、`figures/`

- **文档体系**：`docs/`、`generate_project/**/docs/`、`machine_learning/report/`（文献综述 01–07）

