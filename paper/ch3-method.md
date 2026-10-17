# 3. Method

> 状态：3.3 drafted / 3.4 drafted（骨架）/ 3.1、3.2 stub。写作纪律见 paper/README.md。

## 3.1 问题设定与两域数据概览 — stub

（待写：任务定义——小样本条件眼底生成；A 域 330→现存 299 无标签网络图 / B 域老师 758 四分类两池独立，口径 #7/8；条件 = 血管骨架 vessel skeleton）

## 3.2 FiLM 条件 UNet 架构 — stub

（待写：FiLM 调制 Feature-wise Linear Modulation 逐通道 h×(1+scale)+shift、zero_module、EMA 0.9999、UNet 86.8M；素材 REPORT_DL_text §5.10、EXPERIMENTS DL-EX-013/014）

## 3.3 损失函数设计：从 MSE 到 L1+LPIPS（loss design）

<!-- 2026-10-17 首批写作。主张：暗背景占绝对多数时 MSE 最优策略是"安全暗色"，设计层预判改 L1 + LPIPS（t<200 权重 0.1），FiLM 提供通道级条件增益。只讲设计动机，不带结果数字（结果在 5.4，前向指针）。证据：REPORT_DL_text §6.3 行 951、§5.10.2 行 899-913；--loss_type 可复现开关。 -->

（待起草）

## 3.4 两层评估指标与六维门控评分（evaluation design）

<!-- 2026-10-17 首批写作（骨架）。主张：通用分布层 + 眼底语义层，动机是防通用指标盲区；六维门控把"低层代理可被博弈"的维度乘上"整体像真实"的门。证据：eval/metrics_common.py、metrics_fundus.py、score_scheme.py（SCHEME 权重）、docs/09；门控前后 VAE 57.9→25.7 实例。注意：与人工分的关系是"校准"而非"等价"（τ=0.73，N 小措辞克制，口径 #2）。 -->

（待起草）
