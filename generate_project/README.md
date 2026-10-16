# generate_project —— 生成方法库（按方法族分区）

> 本项目研究生成重度眼底图（KW-IV 级）扩充分类器训练集。生成方法按技术路线分目录管理，共用根目录的 `eval/`（评估/评分体系）、`eval_data/`（评估数据）、`fundus/`（数据集）。

## 目录结构

| 目录 | 方法族 | 状态 |
|:--|:--|:--|
| [deep_learning/](deep_learning/) | 深度学习从零训练：Diffusion（主力）/ GAN / VAE | ✅ Phase A/B 完成 · 🔴 **Phase D 主力进行中**（D1/D2） |
| [pretrained/](pretrained/) | 预训练模型（Stable Diffusion LoRA 等） | ✅ 调研完成（报告 07）· ⏳ 落地待定 |
| [machine_learning/](machine_learning/) | 传统机器学习 / OpenCV（非深度学习，老师 Bug 2） | ✅ 完成（Phase C2：EX-001~005） |
| [transfer_learning/](transfer_learning/) | 迁移学习 | ✅ 文献归档 |

## 宏观文档（跨方向）

[`docs/`](docs/) —— 宏观设计规范 / 工作说明 / 执行步骤 / 导航（2026-08-03 建立）。

各方向具体文档在其子目录内：
- `machine_learning/`：`docs/`（工程）+ `report/`（文献报告）
- `pretrained/`：`report-pretrained-lit.md`（文献归档）
- `transfer_learning/`：`report-tl-lit.md`（文献归档）
- `deep_learning/`：`README.md` + 各项目 `STANDARDS.md`

## 快速入口

- 宏观规范 → [`docs/00-Index.md`](docs/00-Index.md)
- 生成/评估命令 → 根目录 `docs/08-Work-Guide.md`（操作手册）+ `docs/02-Work-Guide.md`（本库宏观）
- 项目指引/状态 → `research-report/CONTINUE_GUIDE_NEW.md`（v24）
- **实验数字唯一来源** → `deep_learning/EXPERIMENTS.md`（DL-EX）+ `machine_learning/docs/06-Records.md`（EX）
- 评估体系 → `eval/`（metrics_common / metrics_fundus / plot_metrics / score_scheme）
- 综合评分设计 → `docs/09-Score-Scheme-Design.md`
