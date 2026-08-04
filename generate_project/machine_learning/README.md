# machine_learning/ —— 传统机器学习生成方法

> Phase C2（老师 Bug 2：方法不限于深度学习）：用传统 ML / 经典 CV 方法生成"相似但不相同"的重度眼底图（DR KW IV 级），用于扩充 DR 分类器训练集。

**状态**：📌 文献调研完成 + 文档体系建立（2026-08-03）。实验未开始（等待用户审阅文档后启动）。

**边界（重要）**：
- ✅ 这里**只做传统 ML**（非深度学习）生成。
- ⚠️ **已做的深度生成方法**（VAE/DCGAN/4 个 DDPM）评估见 `research-report/evaluation_report.md`，本目录不重复。
- ⚠️ **预训练/迁移学习**（SD LoRA、预训练骨干）见 `generate_project/pretrained/`。

---

## 双导航

### 📚 报告（为什么 / 学到了什么 / 论文怎么写）→ `report/`
| 文档 | 内容 |
|------|------|
| [report/00-README.md](report/00-README.md) | 报告导航 + 边界说明 + 三句话结论 |
| [report/01-literature-trad-ml.md](report/01-literature-trad-ml.md) | 传统 ML 图像生成 18 法（含链接） |
| [report/02-literature-fundus.md](report/02-literature-fundus.md) | 眼底特异生成 + 新奇算法组合（含链接） |
| [report/03-literature-augment.md](report/03-literature-augment.md) | 小样本医学经典增广证据（传统 ML 范畴） |
| [report/04-synthesis-insights.md](report/04-synthesis-insights.md) | ⭐ 综合洞察：经验/灵感/为什么有用/后续做法 |
| [report/05-paper-structure.md](report/05-paper-structure.md) | C2 论文结构规划 |

### 🔧 工程规范（怎么做 / 标准）→ `docs/`
| 文档 | 内容 |
|------|------|
| [docs/00-Index.md](docs/00-Index.md) | 工程文档导航 |
| [docs/01-Requirements.md](docs/01-Requirements.md) | 需求规格（相似但不相同，规模化约束） |
| [docs/02-Technical-Standards.md](docs/02-Technical-Standards.md) | 技术规范（数据口径/依赖/CLI/输出/命名） |
| [docs/03-Design-Spec.md](docs/03-Design-Spec.md) | 设计规范（结构/纹理/融合/多样化 分层管线） |
| [docs/04-Work-Guide.md](docs/04-Work-Guide.md) | 工作说明（常用命令/已知坑/评估接入） |
| [docs/05-Execution-Plan.md](docs/05-Execution-Plan.md) | 执行步骤（Phase C2 实验，含质检门控） |
| [docs/06-Records.md](docs/06-Records.md) | 实验记录规范（EX-XXX 模板） |

### 📜 代码 → `scripts/`
| 脚本 | 方法 | 状态 | 总分(六维门控) |
|------|------|:--:|--:|
| [scripts/pca_gen.py](scripts/pca_gen.py) | PCA 线性重建（确定性基线） | ✅ 已完成 | 20.0 |
| [scripts/gmm_gen.py](scripts/gmm_gen.py) | GMM 混合采样（确定性基线） | ✅ 已完成 | 40.2 |
| [scripts/patch_gen.py](scripts/patch_gen.py) | 补丁拼接（确定性基线） | ✅ 已完成 | 42.1 |
| `scripts/poisson_gen.py` | 泊松病变重排（高价值组合） | ⏳ 待建 | — |
| `scripts/retinex_gen.py` | Retinex 光照交换（高价值组合） | ⏳ 待建 | — |
| `scripts/memory_check.py` | 复制检测三件套（质检） | ⏳ 待建 | — |

> 分数口径：**C2ST 必跑**（同口径才与深度方法可比）。3 基线已写入 `research-report/REPORT_ML.docx`（`_build_report_ml.py` 生成）。
> 深度最佳对照：film_l1lpips 70.8（同口径）。

---

## 快速开始

```bash
# 冒烟测试（先 5 张）
cd generate_project/machine_learning
python scripts/pca_gen.py --num_images 5

# 全量生成（各 300 张）
python scripts/pca_gen.py --num_images 300 --seed 42
python scripts/gmm_gen.py --num_images 300 --seed 42
python scripts/patch_gen.py --num_images 300 --seed 42

# 评估（项目根目录）+ 打分入图
cd d:/AI_Model_Project_for_Fundus_Color_Images
python eval/metrics_common.py --real eval_data/real --fake eval_data/pca/singles --img_size 128 --device cuda --json
python eval/score_scheme.py --scorecard && python eval/plot_metrics.py
```

完整步骤见 [docs/05-Execution-Plan.md](docs/05-Execution-Plan.md)。

**参考**：根目录 `research-report/CONTINUE_GUIDE_NEW.md`（全局进度）+ `docs/09-Score-Scheme-Design.md`（评分体系）。
