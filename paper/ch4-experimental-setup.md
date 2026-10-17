# 4. Experimental Setup

> 状态：4.1 drafted / 4.3 drafted（半节）/ 4.2、4.4 stub。数据描述一律从 docs/15 取数。

## 4.1 数据来源与数据卫生（data sources and hygiene）

**两池独立，永不混写。** 本研究涉及两个相互独立的数据池。**A 域**：网络收集的重度（KW-IV）眼底彩照 330 张（无标签、无病灶 mask），作者逐张质检后手动淘汰 31 张，现存 299 张（`fundus/_all_images_ORIGINAL`）；其中作者认可"过关"的手挑池 163 张（池内去重后 153 唯一）。Phase A 全部评估使用 2026-07 冻结的 330 张快照（`eval_data/real`）——该快照含后来被淘汰的 31 张，属**历史冻结口径**，与全部历史指标可比；是否重建更干净的 real 集属写作期决策，若重建须声明口径变更。**B 域**：合作导师数据集 `HRSG_4_758_v0.3.4`，共 758 张（train 552 / val 135 / test 71），normal/mild/moderate/severe 四分类，官方 split 只读。

**severe 类的真实规模。** B 域 train 的 severe 文件数为 152，但其中 76 张是数据集自带的垂直翻转副本（`_aug_1`）——按唯一底图计，severe 只有 **76 张**，是 train 里最少的类。"train 里 severe 已最多（152）"是文件数假象。severe 类的原始分辨率也系统性偏低（train 中位 2048×1536，normal 为 3456×3456）。

**生成训练池与数据卫生三查。** 生成器训练池 = A 域手挑 153 唯一 + B 域 severe 67 唯一，经两批清理（239→229→193，逐张原因见墓园 `fundus/_removed_dups_sq256/MANIFEST.md`，可回滚）。清理依据三连查（工具 `fundus/find_rotation_copies.py`，全部肉眼复核）：

1. **旋转副本：0。** 163 张手挑池经 15° 步进全角度 × 翻转扫描零命中。方法论教训：v1 以 64px 灰度 + 松阈值误报 74 对（把"深色眼底长得像"判成旋转），破绽是同一对在 60/120/180° 同时"匹配"——真旋转只在唯一角度匹配。
2. **池内重复：网络 163→153 唯一；severe 76→67 唯一**（如 80026≡80054）；**跨池同图 16 对**（两池都源自公开数据，如 28_left≡CASE_002）。
3. **跨 split 泄漏（最严重发现）。** B 域官方 split 的 train↔val/test 存在像素级同图：severe 3 对 MSE=0（80024(val)≡80052(train) 等）、moderate 1 对（16794(val)≡16788/16796(train)）、网络图撞 val/test 5 对。全树审计（758 张 × 四类 × 三 split，阈值 0.0026）共 **113 对**相似，大量相邻编号连拍序列被拆进不同 split——数据集按"图"而非按"眼/序列"划分。

**处置与后果。** 泄漏文件已从 train 侧生成池移除；val/test 未动（官方数据集不做任何改动）。后果：下游分类器基线 QWK 0.747@256 含乐观偏差，本文将其作为 dataset limitation 记录（口径 #15），并已提请导师注意（建议按眼/序列重划分，是否执行由导师决定）。硬约束：生成器训练只用 train 侧数据，test 保持全真实。

## 4.2 生成模型训练配置 — stub

（待写：11 方法超参总表；深度 6 方法素材 EXPERIMENTS.md 各条目；传统 5 方法素材 machine_learning/EXPERIMENTS.md）

## 4.3 评估协议（evaluation protocol）

**样本量与采样。** 11 个方法的横评库（6 深度 + 5 传统 ML）使用历史评估：每模型 300 张、128×128、真实基准 330 张（A 域冻结快照，见 4.1）。2026-08-18 之后新增的方法统一 `--num_images 60`；正文报告处均注明样本量。采样器统一 DDIM 50 步（50 步与 100 步实测已收敛，`eval_data/_step_compare/`）；B 域出图判定用 100 步。评估采样固定 seed 42。

**出图判定纪律。** 凡以生成图判断成败，一律使用原始权重出图（`--use_raw`）：EMA 权重在续训过的 checkpoint 上可能退化，不可作为判定依据（DL-EX-015 排查中确认）。

**两套评分尺度，永不混排。** ①**人工分**（human score，0-100）：单人目测校准，校准集 N=6（film_l1lpips 87.5 / film 85 / base_cj 75 / cond 70 / dcgan 20 / vae 10，`eval/human_scores.csv`）；只在校准语境出现并注明"单人 N=6"。②**六维门控总分**（gated composite score，0-100）：3.4 所述六维加权 + 门控，横评库 11 模型现版（`eval_data/_scores.json`，formula v1，n_models=11）。该总分为 min-max **相对分**，模型库扩大会整体平移——引用必须注明版本；本文主用②，①仅用于校准分析。

## 4.4 下游验证协议（TSTR）— stub + 【占位:TSTR-下游验证】

（待写：只补最缺类（severe）、真实 test 评估、相对 TRTR 基线降幅；基线 QWK 0.747@256 含跨 split 泄漏乐观偏差（口径 #15），置换前后不可比；协议出处 classify_project/README.md + Esteban 2017（R6））
