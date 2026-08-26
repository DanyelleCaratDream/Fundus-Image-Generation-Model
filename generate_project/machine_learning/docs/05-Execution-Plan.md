# 执行步骤：Phase C2 传统 ML 实验

> 更新日期：2026-08-03（v2：阶段 1 已完成 + C2ST 必跑 + 六段式汇报）
> 目标：3 个确定性基线 + 2 个高价值组合，各 60 张（2026-08-18 新规，不再 300 全量）→ 评估 → 打分入图 → 报告。
> **📋 工作方式（强制）**：每个实验全程向用户汇报，六段式 = **是什么 → 怎么做（略写）→ 为什么 → 结果（图片/脚本评分/人眼评分【待人工评估】）→ 致命缺点 → 下一步**。做完先问用户补充/修改，**用户确认该实验 OK 后才进行下一步**。

---

## 阶段 0：前置确认

- [ ] 用户已确认本计划（plan 获批）
- [ ] GPU 可用（`torch.cuda.is_available()`）
- [ ] `fundus/_all_images_ORIGINAL/` 存在（≥300 张）
- [ ] `eval_data/real/` 330 张真实评估图已就位

## 阶段 1：确定性基线（3 个，先跑）

### 步骤 1.1 重写基线脚本（纳入 docs/02 规范）
- [ ] `scripts/pca_gen.py`：PCA 线性重建（`--k` 主成分数，默认 64）
- [ ] `scripts/gmm_gen.py`：GMM 混合采样（`--pca_dim` 先降维 + `--n_components` K，默认 16）
- [ ] `scripts/patch_gen.py`：补丁拼接（`--patch_size`，默认 64）

### 步骤 1.2 冒烟测试
```bash
cd generate_project/machine_learning
python scripts/pca_gen.py   --num_images 5
python scripts/gmm_gen.py   --num_images 5
python scripts/patch_gen.py --num_images 5
```
- [ ] 每脚本 5 张产出，`../../eval_data/{pca,gmm,patch}/singles/` 有 `sample_0000.png`
- [ ] 抽查 1 张尺寸/颜色正常（用 PIL 检查 128×128 RGB）

### 步骤 1.3 标准生成（各 60 张，2026-08-18 新规不再 300）
```bash
python scripts/pca_gen.py   --num_images 60 --seed 42
python scripts/gmm_gen.py   --num_images 60 --seed 42
python scripts/patch_gen.py --num_images 60 --seed 42
```
- [ ] 3 × 60 张产出，命名 `sample_0000.png`~`sample_0059.png`

### 步骤 1.4 通用层评估（根目录）
```bash
cd d:/AI_Model_Project_for_Fundus_Color_Images
python eval/metrics_common.py --real eval_data/real --fake eval_data/pca/singles   --img_size 128 --device cuda --json
python eval/metrics_common.py --real eval_data/real --fake eval_data/gmm/singles   --img_size 128 --device cuda --json
python eval/metrics_common.py --real eval_data/real --fake eval_data/patch/singles --img_size 128 --device cuda --json
```
- [ ] `eval_data/{pca,gmm,patch}_metrics.json` 产出

### 步骤 1.5 专用层评估（**C2ST 必跑，不跳过**）
> ⚠️ v1 曾写"跳过 C2ST"导致 patch 62.9 伪高分（D2 口径不均，见 report 2.7 踩坑教训）。**C2ST 是 D2 权重 0.40 的成员，跳过会使 ML 与深度模型不可比。禁止 `--skip_c2st`。**（无条件图仅跳过 cond_path 血管 dice）
```bash
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/pca/singles   --model pca   --device cuda --json
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/gmm/singles   --model gmm   --device cuda --json
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/patch/singles --model patch --device cuda --json
```
- [ ] `eval_data/{pca,gmm,patch}_fundus_metrics.json` 产出（含 c2st_auc）

### 步骤 1.6 复制检测三件套（质检门控）
- [ ] `scripts/memory_check.py` 建立并跑 3 个基线
- [ ] 记录 NN 距离分布 / 全图复制率 / 补丁重复率
- [ ] **门控**：任一方法复制率过高 → 停下排查，不继续

### 步骤 1.7 打分入图
- [ ] `eval/plot_metrics.py` 的 `MODEL_LABELS` 补 pca/gmm/patch 显示名
- [ ] `python eval/score_scheme.py --scorecard && python eval/plot_metrics.py`
- [ ] `_scores.json` 出现 3 个新行；雷达图/总分表更新

### 步骤 1.8 记录 + 初步结论
- [x] 填 `docs/06-Records.md`（EX-001/002/003，已按最终分数更新）
- [x] **阶段 1 实际结果（2026-08-03）**：PCA **20.0** / GMM **40.2** / 补丁 **42.1**（C2ST 补跑后全 9 模型同口径）
- [x] ⚠️ **门控事件已闭环**：patch 初测 62.9 → 根因 C2ST 缺失（步骤 1.5 已改为必跑）→ 补跑后 42.1。报告第 2.7 节记录此教训。
- [x] 复制检测三件套全部通过（mem_ssim_pct_gt_085=0）
- [x] **六段式汇报**：阶段 1 已向用户汇报（是什么/怎么做/为什么/结果/致命缺点/下一步），等用户确认后进入阶段 2

---

## 阶段 2：高价值组合（2 个，文献支撑最强）

### 步骤 2.1 建组合脚本
- [ ] `scripts/poisson_gen.py`：泊松病变重排（cv2.seamlessClone；病变模板跨图采样，防局部复制）
- [ ] `scripts/retinex_gen.py`：Retinex 光照交换（光照×反射分解，跨图交换 L 保留 R）

### 步骤 2.2-2.8 同阶段 1（冒烟 → 全量 → 评估 → 质检 → 打分 → 记录）
- [ ] 各 60 张（新规）→ `eval_data/{poisson,retinex}/singles/`
- [ ] 评估（**含 C2ST**）+ 复制检测（组合方法**必须**重点验"相似但不相同"）
- [ ] `MODEL_LABELS` 补显示名 → score_scheme + plot_metrics
- [ ] 填 `docs/06-Records.md`（EX-004/005）
- [ ] **六段式汇报**：阶段 2 每步向用户汇报，用户确认后才继续

---

## 阶段 3：报告与收尾

### 步骤 3.1 报告 C2 章节
- [ ] 按 `report/05-paper-structure.md` 结构，把实验数据填充进章节
- [ ] 横向对比表：3 基线 + 2 组合 + 6 深度方法统一评分
- [ ] 诚实结论：纯传统 ML 否定性 + 组合价值（新病灶组合）+ 局限声明
- [ ] **REPORT_ML.docx**（research-report/，`_build_report_ml.py` 生成）：阶段 2 完成后重跑脚本，第 7 章替换【待实验】为实际结果；按六段式补充每方法章

### 步骤 3.2 更新文档
- [ ] `report/00-README.md` 状态更新
- [ ] `docs/04-Work-Guide.md` 命令与实际一致
- [ ] 根目录 `research-report/CONTINUE_GUIDE_NEW.md` 更新到 v18

### 步骤 3.3 提交
```bash
git add -A
git commit -m "feat: Phase C2 传统ML实验（3基线+2组合）+ 文档体系"
git push github main && git push gitee main
```

---

## 门控与风险

| 门控 | 触发条件 | 动作 |
|:--|:--|:--|
| 复制率过高 | 任一方法复制检测超标 | 停下，排查（可能是补丁法局部复制），与用户确认 |
| 意外高分 | 某组合得分接近深度最佳 | 停下，与用户确认（是否真有价值/是否伪影），不擅自扩展 |
| 评估异常 | metrics 报错/NaN | 排查数据口径（128×128/命名/数量），修后重跑 |
| 显存 | 评估脚本走 cuda，8GB | 串行跑（不并行 3 个），必要时 `--skip_c2st` |
