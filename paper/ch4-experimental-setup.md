# 4. Experimental Setup

> 状态：4.1 drafted / 4.3 drafted（半节）/ 4.2、4.4 stub。数据描述一律从 docs/15 取数。

## 4.1 数据来源与数据卫生（data sources and hygiene）

<!-- 2026-10-17 首批写作。主张：两池独立（A 域现存 299、评估快照 330 历史冻结；B 域 758 = 552/135/71）；生成训练池 239→193 三查：旋转 0、池内重复（76→67 唯一）、跨 split 泄漏（113 对相似、MSE=0 硬泄漏）。证据：docs/15 §一/§二；口径 #7/8/9/10/11/15。禁写 745；193 与 239 不混；severe 唯一底图 76。 -->

（待起草）

## 4.2 生成模型训练配置 — stub

（待写：11 方法超参总表；深度 6 方法素材 EXPERIMENTS.md 各条目；传统 5 方法素材 machine_learning/EXPERIMENTS.md）

## 4.3 评估协议（evaluation protocol）

<!-- 2026-10-17 首批写作（半节）。主张：统一 --num_images 60（新方法）/历史 300 张（注明样本量）；DDIM 50 步评估/100 步 B 域出图（50/100 已收敛，eval_data/_step_compare/）；两套评分尺度定义与分工。证据：口径 #2/14/16；human_scores.csv（单人 N=6）；_scores.json（formula v1，n_models 11）。 -->

（待起草）

## 4.4 下游验证协议（TSTR）— stub + 【占位:TSTR-下游验证】

（待写：只补最缺类（severe）、真实 test 评估、相对 TRTR 基线降幅；基线 QWK 0.747@256 含跨 split 泄漏乐观偏差（口径 #15），置换前后不可比；协议出处 classify_project/README.md + Esteban 2017（R6））
