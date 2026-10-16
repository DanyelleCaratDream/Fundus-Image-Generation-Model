# 数据文档（Data Documentation）

> 建立于 2026-10-16。全部数据池的**权威描述**：路径、数量、来源、用途、卫生状态。
> 写论文 Experimental Setup 从这里取数；数据有变动先改本文件。
> 配套：卫生工具 `fundus/find_rotation_copies.py` ｜ 墓园 `fundus/_removed_dups_sq256/`（含 MANIFEST.md，可回滚）

---

## 一、数据池总表

### A 域 —— 网络下载图（生成器训练 + 评估 real 基准）

| 池 | 路径 | 数量 | 说明 | 状态 |
|:--|:--|--:|:--|:--|
| A 域训练基准 | `fundus/_all_images_ORIGINAL/` | **299** | 方形 JPG，文件名前缀 `HRSRC_`（ATLAS_/CASE_/DATASET_ 等） | ⚠️ 作者注明：**质量参差**（见下方层级） |
| 评估 real 快照 | `eval_data/real/` | **330** | Phase A 评估时的基准集（改名 real_0000…，resize 128） | ✅ 冻结，勿动 |
| **作者认可池** | `fundus/all_ORIGINAL_sq256/` | **163** | 作者逐张手挑的**"过关"集合**（池内 10 张重复/镜像 → 实际 153 唯一） | ✅ 生成器数据的优先来源 |
| 旧派生集 | `fundus/_all_images_raw / _all_images_256 / _all_images_much` | — | 1320 = 330×4 每图 4 几何变体（用户记忆为 60° 步进旋转） | 🔴 **已不在磁盘**（10 月瘦身移除，归档目录亦无备份）——旧文档仍在引用，本行即更正 |

> **A 域质量层级（作者 2026-10-16 口述确认）**：
> `all_ORIGINAL_sq256`（163，作者认可的过关图）＞ `_all_images_ORIGINAL`（299，含部分不过关图）＞ 历史口径 330（含作者后来手动删除的 31 张质量不过关图）。
> **含义**：① 299 vs 330 之谜已解——31 张为作者手动质量淘汰；② `eval_data/real`（330）快照里含有作者已判"不过关"的图，作为评估基准是**历史冻结**口径（与全部历史 FID/指标可比），论文如需更干净的 real 集须重建并声明口径变更（写作时决策，见 §四）。

> A 域全部为重度（KW-IV）眼底彩照，**无标签、无病灶 mask**；血管骨架 330 张在 `generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/`。

### B 域 —— 老师数据集（DR 四分类，Phase D 主战场）

| 池 | 路径 | 数量 | 说明 | 状态 |
|:--|:--|--:|:--|:--|
| 官方数据集 | `fundus/HRSG_4_758_v0.3.4/` | **758** | train 552 / val 135 / test 71，normal/mild/moderate/severe 四类，全 jpg | 🔒 **只读**——官方 split 不做任何改动 |
| 全集方图 | `fundus/HRSG_4_758_square/` | 758（3 split） | 居中裁方派生 | 派生 |
| 全集 256 | `fundus/HRSG_4_758_sq256/` | 758（3 split） | 256×256 派生（分类器用） | 派生 |
| severe 训练池 | `fundus/severe_train_sq256/` + `severe_train_sq128/` | 各 76 | train severe 唯一底图（DL-EX-015/016/018 用） | ✅ |
| 合并训练池 | `fundus/combined_sq256/` | **193** | 网络手挑 153 唯一 + 老师 severe 67 唯一（去重去泄漏后）；条件 mask `conditions_combined` 193 张一一对应 | ✅ D1/D2 用 |

**老师数据集逐类计数**（文件数；severe 的 152 含 76 张数据集自带垂直翻转 `_aug_1` 副本）：

| split | normal | mild | moderate | severe（文件） | severe（唯一底图） |
|:--|--:|--:|--:|--:|--:|
| train | 120 | 140 | 140 | 152 | **76** |
| val | 34 | 40 | 40 | 21 | 21 |
| test | 18 | 20 | 20 | 13 | 13 |

> ⚠️ severe 类分辨率系统性偏低（train 中位 2048×1536，normal 为 3456×3456）；全集无一张原生 512×512。
> "train 里 severe 已最多（152）"是**文件数假象**——按唯一图算 severe 是 train 里**最少**的类，与"只对它做过增广"互为印证。

### 墓园（graveyard，先归档不销毁）

| 位置 | 内容 |
|:--|:--|
| `fundus/_removed_dups_sq256/` | combined_sq256 两批清理移出的 36 图 + 36 mask，按原子目录存放；`MANIFEST.md` 逐条记原因，**可回滚** |

---

## 二、数据卫生结论（2026-10-16 三连查 + 全树审计）

工具：`fundus/find_rotation_copies.py` v2（RGB 128px、中心圆盘 mask、15° 步进全角度 × 翻转两支、角度互斥判据、分位数校准）。**全部结果经肉眼复核**（`fundus/_dup_check_pairs.png`、`_leak_check_pairs.png`）。

### 2.1 旋转副本：0

`all_ORIGINAL_sq256`（163 张）全角度扫描零命中。历史几何增强只存在于旧派生集 `_all_images_256/_much`（已移出磁盘）。
> 方法论教训（v1 误报 74 对）：64px 灰度 + 松阈值把"深色眼底长得像"判成旋转；破绽是同一对在 60/120/180 三角度同时"匹配"——**真旋转只在唯一角度匹配**。v2 加 RGB/128px/角度互斥后归零。

### 2.2 池内重复

- 网络手挑 163 张 → 10 张重复/镜像 → **153 唯一**（如 CASE_041≡CASE_043）
- 老师 severe 76 张 → 9 张内部重复 → **67 唯一**（如 `80026≡80054`、`80027≡htn_retinopathy`）
- 跨池同图 16 对（两池都源自公开数据，如 `28_left≡CASE_002`）

### 2.3 跨 split 泄漏（最严重发现）

老师官方 split 的 train↔val/test 存在**像素级同图**：
- severe：`80024(val)≡80052(train)`、`80029(val)≡80038(train)`、`80028(test)≡80055(train)`（均 MSE=0）
- moderate：`16794(val)≡16788/16796(train)`
- 另有网络图撞 val/test 5 对（`139_right≡CASE_012`、`28_right≡CASE_003`、`239904≡ATLAS_155_A`、`861_right≡CASE_039`、`1004c816≡ATLAS_152_A`）+ 连拍/翻转近重复若干

**全树审计**（758 张 × 四类 × 三 split，0°+flip，阈值 0.0026）：跨 split 相似对 **113 对**；大量相邻编号连拍序列被拆进不同 split → **数据集按"图"而非按"眼/序列"划分**。

**处置**：泄漏文件 16 张已从 combined_sq256 移除（train 侧）；val/test 未动（官方数据集不改）。**后果**：分类器基线 QWK 0.747@256 含乐观偏差 → 论文作为 dataset limitation 记录。

### 2.4 📣 需转告老师两件事

1. 官方 split 存在跨 split 同图（上列清单），当前基线指标含乐观偏差；
2. 建议按**眼/序列**分组重新划分（是否重划由老师决定；我们不动官方数据）。

---

## 三、combined_sq256 的构成（D1/D2 / 正式生成训练集）

```
combined_sq256（193）
├── 网络手挑（all_ORIGINAL_sq256 → 去重）      153
└── 老师 severe train（去内部重复）             67
    （去跨池同图 16 对 → 见 MANIFEST 两批清理 239→229→193）
```

- 条件 mask：`generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions_combined/`（193，stem 一一对应）
- 网络部分源自 `all_ORIGINAL_sq256`（作者认可的**过关**集合，见 §一质量层级）
- 硬约束：**只用 train 侧数据**；val/test 图不入池（防泄漏，见 classify_project/README.md）
- 逐张清理原因见 `fundus/_removed_dups_sq256/MANIFEST.md`

---

## 四、未决事项

| # | 事项 | 状态 |
|:--|:--|:--|
| 1 | ~~299 vs 330 差异~~ **已解决（2026-10-16 作者确认）**：差的 31 张是作者手动质量淘汰。遗留子问题：`eval_data/real`（330）快照含已淘汰图，是否重建更干净的 real 集 = **论文写作时决策**（重建会破坏与历史 FID 的可比性，须声明口径） | ✅ 已答 / ⏳ 写作时决策 |
| 2 | `_all_images_raw/_256/_much` 三派生集已彻底离开磁盘（连归档目录都没有）——若论文需要复现旧训练集，只能从 `_all_images_ORIGINAL` + 增强脚本重建 | ⚠️ 已记录 |
| 3 | 转告老师两件事（2.4 节） | ⏳ 待传达 |
| 4 | C2ST@256 检查 | ⏳ 未排期 |
