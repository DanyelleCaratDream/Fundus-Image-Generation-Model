# 5. Results — 失效机理链（failure mechanism chain）

> 状态：5.2–5.6 首批起草中，5.1 收尾写，5.7 stub。**数字放行前三点头核验**（EXPERIMENTS.md 汇总行 → 详细段 → eval_data JSON）。

## 5.1 失效机理链总览

<!-- 最后写：一图一表串六环，每环带证据编号。图 1 占位。 -->

（待起草）

## 5.2 VAE：模糊天花板（blur ceiling）

<!-- 主张：像素级 MSE 重建 + 小样本 → 模糊是天性，加容量/轮数不可逃逸。证据：REPORT_DL_text §3.2 行 294-353（"平均脸"比喻保留）、§3.4-3.6（800 vs 1200 轮肉眼无差）；EX-001/002/003；vae_metrics.json（FID 188.4 但 C2ST AUC 0.9992 = 颜色假象）、vae_fundus_metrics.json（hemo_retention 0.6997）；门控 57.9→25.7（口径 #2）。DL-EX-001/002/003。 -->

（待起草）

## 5.3 GAN：判别器记忆（discriminator memorization）

<!-- 主张：三变体同根因——330 张判上千遍，判别器背题后梯度消失或 Critic 分数爆炸。证据：REPORT_DL_text §4.2-4.12、§4.10 三行根因表（行 502-516，"背题/跟空气博弈"比喻保留）；DL-EX-005（FID 229.8，口径 #4 轮数 1500）/006（2 轮 D −398.89）/007（loss 稳但无细节）/008（环境）；dcgan_fundus_metrics.json（hemo_retention 0.0、C2ST 1.0、exud_retention 40.0）。新术语：判别器记忆 discriminator memorization（回写 docs/14 §四）。 -->

（待起草）

## 5.4 MSE 病灶溶解：受控消融（lesion dissolution）

<!-- 核心，先写。主张：MSE 损失下出血保留率 0.254（丢 76%），换 L1+LPIPS 恢复 1.137（4.5×）；Palette 异架构复现同现象 → 归因损失而非架构。证据：REPORT_DL_text §5.9.4 行 842-853（epoch 220/500/800"细节先好后差"）、§5.9.5-5.9.6（Palette 同效应+推理完全无效）、§6.3 行 951；DL-EX-011/012/013 vs 014；film_fundus_metrics.json（0.2539、C2ST 0.9892、hemo_presence 0.22）vs film_l1lpips_fundus_metrics.json（1.1368、C2ST 0.9146 全场最低）。指标口径：仅出血（hemo），渗出（exud）超冲 9.6/13.7 如实注明不用于主张。 -->

（待起草）

## 5.5 L1+LPIPS 修复效果与双尺度评分

<!-- 主张：出血保留率 4.5× 恢复 + FID 178.8（vs MSE 版 200.2）；人工分 87.5（human，单人 N=6）与门控分 45.2（gated，v1/11 模型）并存 → 相对分语义引出 6.2。证据：REPORT_DL_text §5.10.2 行 899-913（白点伪影/边缘噪点如实写，行 911-912）；human_scores.csv；_scores.json；纪律 #6 干净样本率 25-30%。禁 72.9/44.2 历史分。 -->

（待起草）

## 5.6 数据量假说与单变量验证（data quantity hypothesis）

<!-- 主张：76 张 loss 正常收敛（0.05-0.08，与成功配方 0.052 同量级）但生成塌缩——"拟合≠生成"；016 排除分辨率；018 微调保留能力（干净 2/8 与 014 常态持平）但失败样本崩得更狠；D1 失败（0/8）且归因混杂（bd64 ¼ 容量 + 256px 4× 双变量）→ 单变量原则教训。证据：EXPERIMENTS.md DL-EX-015 行 78-128 / 016 行 148-164 / 017 行 169-170 / 018 行 177-195 / 019 行 251-262；docs/15。表述边界：可写"训练目标可收敛，但采样结果塌缩，机理归因待单变量验证"，禁写"证明无泛化能力"（N=8 小、D1 混杂未解除）。占位：D3/D4 裁决，禁预写结论。 -->

（待起草，含【占位:D3-数据量裁决】【占位:D4-微调vs从零】）

## 5.7 传统 ML 基线与组合横评 — stub

（**今晚不写**。素材：`machine_learning/EXPERIMENTS.md` EX-001~005 + REPORT_ML + `machine_learning/report/05-paper-structure.md`；门控分（gated，v1）poisson 80.8 / retinex 72.5 / patch 31.0 / gmm 22.7 / pca 11.9；核心论点 = 纯传统 ML 不可行但组合管线有补充价值，回应"不用深度学习行不行"）
