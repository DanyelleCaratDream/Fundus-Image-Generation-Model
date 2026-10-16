# machine_learning/ —— 传统机器学习生成方法

> Phase C2（老师 Bug 2：方法不限于深度学习）：用传统 ML / 经典 CV 方法生成"相似但不相同"的重度眼底图（DR KW IV 级），用于扩充 DR 分类器训练集。

**状态**：✅ **Phase C2 全部完成**（2026-08-26：3 基线 + 2 组合方法，EX-001~005，REPORT_ML.docx 闭环）。
文档体系 2026-10-16 收敛：方法级 `docs/` 已取消（工程规范统一在根 `docs/`），实验登记升格为本目录 `EXPERIMENTS.md`。

**边界（重要）**：
- ✅ 这里**只做传统 ML**（非深度学习）生成。
- ⚠️ **已做的深度生成方法**（VAE/DCGAN/4 个 DDPM）评估见 `research-report/evaluation_report.md`，本目录不重复。
- ⚠️ **预训练/迁移学习**（SD LoRA、预训练骨干）见 `generate_project/pretrained/`（调研报告 07 在 `report/07-pretrained-gen-models.md`）。

---

## 导航

### 📜 实验登记 → [EXPERIMENTS.md](EXPERIMENTS.md)
EX-001~005 全记录（配置/命令/六段式/失败分析），**数字唯一来源**。传统 ML 编号 `EX-xxx` 与深度学习 `DL-EX-xxx` 区分。

### 📚 报告（为什么 / 学到了什么 / 论文怎么写）→ `report/`
| 文档 | 内容 |
|------|------|
| [report/00-README.md](report/00-README.md) | 报告导航 + 边界说明 + 三句话结论 |
| [report/01-literature-trad-ml.md](report/01-literature-trad-ml.md) | 传统 ML 图像生成 18 法（含链接） |
| [report/02-literature-fundus.md](report/02-literature-fundus.md) | 眼底特异生成 + 新奇算法组合（含链接） |
| [report/03-literature-augment.md](report/03-literature-augment.md) | 小样本医学经典增广证据（传统 ML 范畴） |
| [report/04-synthesis-insights.md](report/04-synthesis-insights.md) | ⭐ 综合洞察：经验/灵感/为什么有用/后续做法 |
| [report/05-paper-structure.md](report/05-paper-structure.md) | C2 论文结构规划 |
| [report/06-defense-qa.md](report/06-defense-qa.md) | 答辩防御手册（约 30 问） |
| [report/07-pretrained-gen-models.md](report/07-pretrained-gen-models.md) | ⭐ 预训练生成模型调研（通用→眼底，SD-LoRA 主路线） |

### 🧰 代码 → `scripts/`
| 脚本 | 方法 | 六维总分（11 模型现版） |
|------|------|--:|
| [scripts/pca_gen.py](scripts/pca_gen.py) | PCA 线性重建（确定性基线） | 11.9 |
| [scripts/gmm_gen.py](scripts/gmm_gen.py) | GMM 混合采样（确定性基线） | 22.7 |
| [scripts/patch_gen.py](scripts/patch_gen.py) | 补丁拼接（确定性基线） | 31.0 |
| [scripts/poisson_gen.py](scripts/poisson_gen.py) | 泊松贴片重排（高价值组合，v6） | **80.8** |
| [scripts/retinex_gen.py](scripts/retinex_gen.py) | Retinex 光照交换（高价值组合） | **72.5** |
| `scripts/memory_check.py` | 复制检测三件套（质检） | ⏳ 待建（现由评估内记忆检测覆盖） |

> ⚠️ **min-max 相对分数**：分数随模型库扩大整体平移（film_l1lpips 70.8→47.4→45.2，poisson 87.5→80.8）；上表为**当前 11 模型版**，完成时分数见 EXPERIMENTS.md 各 EX 详节。分数口径：**C2ST 必跑**（同口径才可比）。
> poisson/retinex 高分部分源于底图复用（用户确认有效），REPORT_ML 与 EXPERIMENTS.md 已如实标注。

---

## 快速开始

```bash
# 标准生成（60 张，⚠️ 2026-08-18 新规：不再 300 全量）
cd generate_project/machine_learning
python scripts/pca_gen.py --num_images 60 --seed 42

# 评估（项目根目录）+ 打分入图
cd d:/AI_Model_Project_for_Fundus_Color_Images
python eval/metrics_common.py --real eval_data/real --fake eval_data/pca/singles --img_size 128 --device cuda --json
python eval/score_scheme.py --scorecard && python eval/plot_metrics.py
```

**参考**：根目录 `docs/09-Score-Scheme-Design.md`（评分体系设计）+ `docs/14-Paper-Spine.md`（口径表）+ `research-report/CONTINUE_GUIDE_NEW.md`（全局进度）。
