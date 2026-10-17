# 论文图规格约定 + 占位清单

> 依据 docs/14 写作纪律 #3。今晚（2026-10-17）不产图，只登记占位。

## 规格（从出图那一刻就按论文标准）

- **≥300 DPI**；子图带 (a)(b)(c) 标号；每张图的**原始单图另存**（合成图之外保留素材）
- 生成样本图 caption **必须注明筛选事实**：film_l1lpips 干净样本实测仅 ~25-30%（DL-EX-014 常态），展示图为手工挑选
- 素材来源三处：`research-report/figures/`（自动出图）、`REPORT_DL.docx` 内嵌图（epoch 序列）、`generate_project/deep_learning/Fundus-Diffusion/ddpm/results_*/images/`

## 占位清单

| 编号 | 位置 | 内容 | 素材 |
|:--|:--|:--|:--|
| 图 1 | 5.1 | 失效机理链总览：六环串图（每环 = 模型 + 死因 + 关键数字） | 新绘（依 scorecard.png 改造） |
| 图 2 | 5.2 | VAE：默认 vs 大容量 800/1200 轮对比（"平均脸"） | REPORT_DL 图 1/2/3 |
| 图 3 | 5.3 | GAN：DCGAN 1500 轮输出（通道塌缩暗红图）+ WGAN-GP 2 轮爆炸曲线 | REPORT_DL 图 4/5 + log |
| 图 4 | 5.4 | MSE 溶解时序：条件扩散 epoch 220/500/800（细节先好后差）+ Palette epoch 125/275 | REPORT_DL 图 10a-c、11a-b |
| 图 5 | 5.5 | L1+LPIPS epoch 700/725/750/775（病灶保持） | REPORT_DL 图 13a-d |
| 图 6 | 5.5 | 双尺度对比：人工分 vs 六维门控分排序差异（87.5/45.2 等） | `eval/human_scores.csv` + `_scores.json` 新绘 |
| 图 7 | 5.6 | 数据量假说实验矩阵与判定面板（D1 0/8） | `D:/tmp/D1_judgment/D1_judgment_panel.png`（需入库）+ D3/D4 出图后补 |
