# 执行步骤：Phase C2 传统 ML 实验

> 更新日期：2026-08-03 ｜ 按本文档顺序执行；每步完成标记 ✅/❌
> 目标：3 个确定性基线 + 2 个高价值组合，各 300 张 → 评估 → 打分入图 → 报告。

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

### 步骤 1.3 全量生成（各 300 张）
```bash
python scripts/pca_gen.py   --num_images 300 --seed 42
python scripts/gmm_gen.py   --num_images 300 --seed 42
python scripts/patch_gen.py --num_images 300 --seed 42
```
- [ ] 3 × 300 张产出，命名 `sample_0000.png`~`sample_0299.png`

### 步骤 1.4 通用层评估（根目录）
```bash
cd d:/AI_Model_Project_for_Fundus_Color_Images
python eval/metrics_common.py --real eval_data/real --fake eval_data/pca/singles   --img_size 128 --device cuda --json
python eval/metrics_common.py --real eval_data/real --fake eval_data/gmm/singles   --img_size 128 --device cuda --json
python eval/metrics_common.py --real eval_data/real --fake eval_data/patch/singles --img_size 128 --device cuda --json
```
- [ ] `eval_data/{pca,gmm,patch}_metrics.json` 产出

### 步骤 1.5 专用层评估（传统 ML 无条件图，跳过 C2ST/血管）
```bash
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/pca/singles   --model pca   --device cuda --skip_c2st --json
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/gmm/singles   --model gmm   --device cuda --skip_c2st --json
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/patch/singles --model patch --device cuda --skip_c2st --json
```
- [ ] `eval_data/{pca,gmm,patch}_fundus_metrics.json` 产出

### 步骤 1.6 复制检测三件套（质检门控）
- [ ] `scripts/memory_check.py` 建立并跑 3 个基线
- [ ] 记录 NN 距离分布 / 全图复制率 / 补丁重复率
- [ ] **门控**：任一方法复制率过高 → 停下排查，不继续

### 步骤 1.7 打分入图
- [ ] `eval/plot_metrics.py` 的 `MODEL_LABELS` 补 pca/gmm/patch 显示名
- [ ] `python eval/score_scheme.py --scorecard && python eval/plot_metrics.py`
- [ ] `_scores.json` 出现 3 个新行；雷达图/总分表更新

### 步骤 1.8 记录 + 初步结论
- [ ] 填 `docs/06-Records.md`（EX-001/002/003）
- [ ] 预期：3 基线得分远低于扩散（0-20 分）→ 否定性结论底成立

---

## 阶段 2：高价值组合（2 个，文献支撑最强）

### 步骤 2.1 建组合脚本
- [ ] `scripts/poisson_gen.py`：泊松病变重排（cv2.seamlessClone；病变模板跨图采样，防局部复制）
- [ ] `scripts/retinex_gen.py`：Retinex 光照交换（光照×反射分解，跨图交换 L 保留 R）

### 步骤 2.2-2.8 同阶段 1（冒烟 → 全量 → 评估 → 质检 → 打分 → 记录）
- [ ] 各 300 张 → `eval_data/{poisson,retinex}/singles/`
- [ ] 评估 + 复制检测（组合方法**必须**重点验"相似但不相同"）
- [ ] `MODEL_LABELS` 补显示名 → score_scheme + plot_metrics
- [ ] 填 `docs/06-Records.md`（EX-004/005）

---

## 阶段 3：报告与收尾

### 步骤 3.1 报告 C2 章节
- [ ] 按 `report/05-paper-structure.md` 结构，把实验数据填充进章节
- [ ] 横向对比表：3 基线 + 2 组合 + 6 深度方法统一评分
- [ ] 诚实结论：纯传统 ML 否定性 + 组合价值（新病灶组合）+ 局限声明

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
