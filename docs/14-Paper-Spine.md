# 论文脊柱（Paper Spine）—— 主张、证据、口径

> 建立于 2026-10-16。这是论文写作的**宪法**：每条主张的证据在哪、每个数字从哪取、所有口径怎么统一。
> 写作前读一遍，写作中随时对照，投稿前逐条自查。
> 卫星文档：[11-Paper-Readiness](11-Paper-Readiness.md)（就绪度诊断）｜[12-Literature-Survey-and-Positioning](12-Literature-Survey-and-Positioning.md)（文献定位）｜[13-Expert-Evaluation-Protocol](13-Expert-Evaluation-Protocol.md)（专家盲评）
> 数字唯一来源：`generate_project/deep_learning/EXPERIMENTS.md`（深度）＋ `generate_project/machine_learning/EXPERIMENTS.md`（传统）＋ `eval_data/*.json`（指标）＋ `eval/human_scores.csv`（人工分）。

---

## 一、路线确认（2026-10-16 用户拍板）

- **路线 A 先行**：《小样本眼底图像生成：失效机理与评估方法》——现有材料即可动笔；
- **路线 B 叠加**：Phase D 下游验证（TSTR）+ 外部对比 + 多 seed → 抬高档次；
- **语言**：中文写作，关键术语随文标注英文（术语对照见第四节），投稿翻译时直接取用。

---

## 二、贡献清单（Claim → Evidence → Number Source）

> 每条主张必须能回指到实验编号。写作时正文不出现"凭空数字"。

### C1 失效机理链条（failure mechanism chain）—— 路线 A 的主轴

小样本眼底生成在三代模型上各有死因，且都能归因到损失/数据机制：

| 环节 | 主张 | 证据 | 数字出处 |
|:--|:--|:--|:--|
| VAE | 像素级 MSE → 模糊天花板（blur ceiling），加大容量/轮数无效 | DL-EX-001/002/003（10 分，800 vs 1200 轮肉眼无差） | EXPERIMENTS.md |
| GAN | 判别器小样本过拟合 → 全面失败 | DL-EX-005（FID 229.8）/006/007（爆炸）/008（环境） | EXPERIMENTS.md |
| Diffusion-MSE | **MSE 溶解病灶**（lesion dissolution）：暗背景 95% 像素摊薄亮病灶梯度 | DL-EX-013 vs 014：出血保留率 **0.254 vs 1.137** | EXPERIMENTS.md + eval_data JSON |
| 跨架构复现 | Palette（完全不同架构）同现溶解 → 是 MSE 的天性非模型缺陷 | DL-EX-012 | EXPERIMENTS.md |
| ⭐ 消融可复现 | `--loss_type {l1,mse}` 开关（2026-10-16 加）→ MSE 臂可一键复现 | train.py + config.json 自动记录 | EXPERIMENTS.md 附 1 |
| 数据量（路线 B 素材） | 76 张 @256/128：loss 正常收敛但生成全糊（"拟合≠生成"） | DL-EX-015/016 ❌；D1/D2（DL-EX-019/020）单变量验证中 | EXPERIMENTS.md |

### C2 评估方法论（evaluation methodology）

| 主张 | 证据 | 注意事项 |
|:--|:--|:--|
| 两层指标体系（通用层 + 眼底专用层）+ 六维门控综合评分 | eval/metrics_common.py + metrics_fundus.py + score_scheme.py；设计动机 docs/09 | τ=0.73 是与人工分校准，N 小，措辞克制 |
| **通用分布指标看不出医学语义差异**：FID 相近但出血保留率差 4.5×（0.254 vs 1.137） | Phase A 专用层 | ⚠️ docs/12：此为已知共识的**新实例**（Deo 2025 等 5 篇），必须引用，不可声称首创 |
| 病灶实例级保留率指标与 LFS（Dice 重分割）/ FRD（分布级）本质不同 | docs/12 第三节（已核实不撞车） | 纪律：不说"和 LFS 一样"，也不说"首创病灶级指标"；引 Asokan 按原文口径（他们没提 FID） |

### C3 同口径多方法横评（unified benchmark）—— 调研确认的空白

11 个方法（5 传统 ML + 6 深度）同一套指标、同一评分口径横向对比。docs/12：调研 30+ 篇全是"提出单方法自证"，无人做统一横评。传统侧证据：`machine_learning/EXPERIMENTS.md`（EX-001~005）+ REPORT_ML。

### C4 下游验证（downstream validation，路线 B）—— 待做

生成 severe 图 → 老师审核替换 76 张翻转副本 → 扩充数据集训分类器 → TSTR 对比基线 QWK 0.747@256。方案：`classify_project/README.md`。**置换前后不可比口径**：基线含跨 split 泄漏乐观偏差（见第五节口径表）。

---

## 三、口径统一表（一次定死，正文只准从这里抄）

> REPORT_DL.docx 的 13 处矛盾大多源于"同一数字多处手写"。本表即终审。

| # | 口径项 | **定论** | 依据 / 说明 |
|:--|:--|:--|:--|
| 1 | 模型类别 | **"三大生成家族（VAE / GAN / Diffusion）＋ 传统 ML 基线与组合"** | 摘要"四类"、结论"三代"、副标题四个（含 Flow Matching）三处打架 → 以此为准；Flow Matching 只出现在展望 |
| 2 | 评分尺度 | **两套，永不混排**：①人工分（human score，0-100，单人目测校准）：film_l1lpips **87.5** / film 85 / base_cj 75 / cond 70 / dcgan 20 / vae 10；②六维门控总分（gated composite score，0-100，11 模型**现版**）：poisson 80.8 / retinex 72.5 / film_l1lpips **45.2** / film 37.2 / cond 35.0 / base_cj 34.5 / patch 31.0 / vae 26.6 / gmm 22.7 / dcgan 12.9 / pca 11.9 | 出处 `eval/human_scores.csv` 与 `eval_data/_scores.json`。论文主用②，①只在校准分析出现并标注"单人 N=6"。⚠️ ②是 min-max **相对分**，模型库扩大会整体平移（film_l1lpips 70.8→47.4→45.2）；引用必须注明 _scores.json 版本，旧文档的 72.9/44.2 等为 6 模型版历史分**不得再用** |
| 3 | 最佳模型称谓 | FiLM DDPM + L1 + LPIPS（film_l1lpips），@128，780 轮，最佳效果 ~700 轮 | 保留 checkpoint 720 对应最佳 |
| 4 | DCGAN 轮数 | **1500** | checkpoint_epoch_001500 为准；报告 4.2 的"800"是笔误 |
| 5 | 基础 DDPM 人工分 | **75** | human_scores.csv；EX-010 的 70 是旧实验 DL-EX-009，EX-011 的 80 未采用 |
| 6 | 实验编号 | 深度 = `DL-EX-xxx`，传统 = `EX-xxx` | 历史上撞过车，正文引用必须带前缀 |
| 7 | 数据池口径 | **A 域**＝网络下载 330 张（无标签，训练生成器+评估 real）；**B 域**＝老师 `HRSG_4_758_v0.3.4`（758 张 = train 552 / val 135 / test 71，四分类） | 两池独立采写，永不混写。注意：旧文档曾写"745"，实数 758 |
| 8 | A 域现状 | `fundus/_all_images_ORIGINAL` 现存 **299** 张（评估快照 `eval_data/real` 为 330 张） | ⚠️ 2026-10-16 盘点发现差 31 张，待作者确认原因；详见 [15-Data-Documentation](15-Data-Documentation.md) |
| 9 | B 域生成训练数据 | `fundus/combined_sq256` = **193** 张（网络 153 唯一 + 老师 67 唯一，去重去泄漏后） | 239→193 两批清理，graveyard 可回滚 |
| 10 | 老师数据集 severe | train severe 唯一底图 **76** 张（另 76 张为数据集自带垂直翻转 `_aug_1`） | "train 里 severe 已最多（152）"是文件数假象，按唯一图 severe 最少 |
| 11 | 几何增强 | 旧派生集 1320 = 330×4（每图 4 变体，已随瘦身移出磁盘）；`all_ORIGINAL_sq256`（163→153 唯一）无旋转副本 | 15° 步进全角度扫描 0 命中（EXPERIMENTS.md 数据卫生节） |
| 12 | Phase D 分辨率 | **256×256**（2026-10-16 用户选定，**取代** 08-26 的"512 原生"决策） | D1/D2 及正式生成均为 256 |
| 13 | Phase D 补图范围 | train + val 补 severe，**test 保持全真实** | TSTR 严谨性 |
| 14 | 评估样本量 | 新方法统一 `--num_images 60`（2026-08-18 起）；历史 300 张不变 | 正文须注明各方法样本量 |
| 15 | 泄漏声明 | 老师官方 split 存在跨 split 同图（severe 3 对 MSE=0 等，全树 113 对相似）→ **基线 QWK 0.747@256 含乐观偏差** | 论文作为 dataset limitation 记录；已列"转告老师"事项 |
| 16 | 采样器 | 评估统一 DDIM 50 步（50/100 实测已收敛，`eval_data/_step_compare/`）；B 域出图判定用 100 步 | |

---

## 四、术语中英对照（随文标注用，持续补充）

| 中文 | 英文 |
|:--|:--|
| 失效机理 | failure mechanism |
| 病灶溶解 | lesion dissolution |
| 出血保留率 | hemorrhage retention ratio |
| 模糊天花板 | blur ceiling |
| 小样本 | few-shot / small-sample |
| 条件生成 | conditional generation |
| 血管骨架 | vessel skeleton |
| 特征线性调制 | Feature-wise Linear Modulation (FiLM) |
| 六维门控评分 | six-dimension gated composite score |
| 训练-合成-测试 | Train on Synthetic, Test on Real (TSTR) |
| 数据泄漏 | data leakage |
| 下游验证 | downstream validation |
| 消融实验 | ablation study |
| 专家盲评 | blinded expert evaluation |

---

## 五、逐节素材指针

| 论文章节 | 素材位置 | 状态 |
|:--|:--|:--|
| Abstract / Title | 本文件第二节贡献清单压缩 | ❌ 待写 |
| 1. Introduction | docs/12 第一节（研究空白）+ REPORT_DL 第 6 章动机改造 | ❌ 待写 |
| 2. Related Work | docs/12 全文 + `machine_learning/report/01-07` + `literature/` | ⚠️ 有料未组织 |
| 3. Method | EXPERIMENTS.md DL-EX-013/014 完整配置 + docs/09（评分设计） | ✅ 最强 |
| 4. Experimental Setup | [15-Data-Documentation](15-Data-Documentation.md)（数据）+ docs/08（命令） | ⚠️ 数据描述待从 15 号取 |
| 5. Results | `eval_data/*.json` + evaluation_report.md + figures | ✅ 硬货 |
| 6. Discussion | docs/13 盲评结果（待做）+ docs/12 的 Low-LFS paradox 对话 + 口径表 #15 泄漏声明 | ⚠️ 部分待实验 |
| 7. Limitations & Conclusion | docs/11 第五节"过度声称高危区"逐条自查 | ⚠️ 需去虚 |

---

## 六、写作纪律（投稿前逐条自查）

1. **数字不手写**：正文数字一律从两份 EXPERIMENTS.md / eval_data JSON 取，改数先改源。
2. **分数必标尺度**：出现任何 0-100 分，括号标注（human / gated）。
3. **图规格**：≥300 DPI、子图 (a)(b)(c) 标号、原始单图另存——从出图那一刻就按论文规格（docs/11 第八节）。
4. **引用纪律**（docs/12）：Deo 2025 必引；Asokan 按原文口径；禁用"首创病灶级指标"类表述。
5. **主张回指**：每条可检索的主张标注证据编号（DL-EX-xxx / EX-xxx / 表号），互查无源即删。
6. **干净样本率口径**：film_l1lpips 实测干净样本仅 ~25-30%，任何"展示图"须手工挑选并注明筛选事实（不可暗示为随机样本）。

---

## 七、缺口跟踪（自 docs/11 第三节，动态更新）

| 缺口 | 状态 | 去向 |
|:--|:--|:--|
| Phase D 下游验证 | 🏃 D1/D2 今晚 | classify_project/README.md |
| 外部/预训练模型对比 | ⏳ 报告 07 已调研（SD-LoRA 主路线） | generate_project/deep_learning 报告 07 |
| 统计显著性（多 seed） | ⏳ 未排期 | 路线 B |
| 多人规范化人工评估 | ⏳ 方案已定 docs/13，评分表模板未做 | docs/13 |
| 患者级划分 | 🔴 已实测确认泄漏（口径 #15），老师决定是否重划 | docs/15 |
| .bib 文献库 | ⏳ 未建 | literature/ |
