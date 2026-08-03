# generate_project 宏观文档导航

> 本目录是 `generate_project/` 的**宏观工程文档**（跨方向）。各方向的具体文档在其子目录内。
> 更新日期：2026-08-03

---

## 文档地图

| 文档 | 内容 |
|------|------|
| [00-Index.md](00-Index.md) | 本文档：宏观导航 |
| [01-Design-Spec.md](01-Design-Spec.md) | 宏观设计规范（方向分工 + 共用评估/数据/评分） |
| [02-Work-Guide.md](02-Work-Guide.md) | 宏观工作说明（跨方向命令速查） |
| [03-Execution-Plan.md](03-Execution-Plan.md) | 宏观执行步骤（Phase C2 → C3 → D 路线） |

## 四方向入口

| 方向 | 文档 | 脚本 |
|:--|:--|:--|
| **deep_learning/**（深度学习从零训练，已完成） | [`deep_learning/README.md`](../deep_learning/README.md) + 各项目 STANDARDS.md | 各项目 train/generate.py |
| **machine_learning/**（传统 ML，Phase C2 当前） | [`machine_learning/docs/00-Index.md`](../machine_learning/docs/00-Index.md)（工程）+ [`machine_learning/report/00-README.md`](../machine_learning/report/00-README.md)（报告） | `machine_learning/scripts/*.py` |
| **pretrained/**（预训练，Phase C3 可选） | [`pretrained/README.md`](../pretrained/README.md) + [`pretrained/report-pretrained-lit.md`](../pretrained/report-pretrained-lit.md) | — |
| **transfer_learning/**（迁移学习，Phase C） | [`transfer_learning/README.md`](../transfer_learning/README.md) + `report-tl-lit.md` | — |

## 共享基础设施（根目录，全方向共用）

| 组件 | 位置 | 说明 |
|:--|:--|:--|
| 评估体系 | `eval/`（metrics_common / metrics_fundus） | 通用层 + 专用层指标 |
| 综合评分 | `eval/score_scheme.py` + `plot_metrics.py` | 六维门控 0-100 总分 + 可视化 |
| 评估数据 | `eval_data/` | real/ + 每模型 300 张 + JSON |
| 数据集 | `fundus/` | `_all_images_ORIGINAL/` 330 张真实图 |
| 工程规范中心 | 根目录 `docs/` | docs/00-10 |
