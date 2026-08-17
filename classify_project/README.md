# classify/ —— DR 分级分类模型（Phase D）

> **方案 1**：与生成项目同仓库（保留记忆/文档/数据共享），独立目录做分类。

## 任务定位
用生成数据（`eval_data/` 各模型 300 张）+ 真实图（`fundus/` 330 张）扩充训练 DR 分级分类器。

## 已确认的约束（老师反馈 2026-08-02）
- **只做分级分类**（不框出/检测病灶，无需 YOLO，纯分类网络）
- **330 张需提前标注等级**（老师/医生标注；本项目 330 张全重度 KW-IV，其他程度在别的同学手里）
- **分级标准 = KW（Keith-Wagener）高血压视网膜病变 I-IV 级**（非 APTOS/ICDR 0-4；本项目生成 IV 级最重图）

## 待办（启动分类时）
- [ ] 330 张标注清单 CSV 模板（标注交付物）
- [ ] 可选简单看图标注工具
- [ ] 等 330 标注到位 + KW 分级细节确认 → 定 Phase C/D 计划
- [ ] TSTR/TRTR 下游验证（终极金标准，见 `docs/09-Score-Scheme-Design.md`）

## 数据/工具入口
- 真实图：根目录 `fundus/_all_images_ORIGINAL/`（330 张）
- 生成图：根目录 `eval_data/{model}/singles/`（每模型 300 张）
- 血管骨架：`generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/`（330 张）
- 评分体系：根目录 `eval/`（score_scheme 可评估生成质量与分类训练集构成）
