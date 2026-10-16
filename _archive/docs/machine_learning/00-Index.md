# machine_learning 工程文档导航

> 本目录是「传统 ML 生成眼底图」的**工程规范中心**（怎么做）。「为什么/学到了什么」见 [`../report/`](../report/)。
> 更新日期：2026-08-03

---

## 文档地图

| 文档 | 内容 | 状态 |
|------|------|:--:|
| [00-Index.md](00-Index.md) | 本文档：工程文档导航 | ✅ |
| [01-Requirements.md](01-Requirements.md) | 需求规格（"相似但不相同"生成，规模化约束） | ✅ |
| [02-Technical-Standards.md](02-Technical-Standards.md) | 技术规范（数据口径/依赖/CLI/输出目录/命名） | ✅ |
| [03-Design-Spec.md](03-Design-Spec.md) | 设计规范（管线架构：结构/纹理/融合/多样化 分层） | ✅ |
| [04-Work-Guide.md](04-Work-Guide.md) | 工作说明（常用命令/已知坑/评估接入） | ✅ |
| [05-Execution-Plan.md](05-Execution-Plan.md) | 执行步骤（Phase C2 实验步骤，含质检门控） | ✅ |
| [06-Records.md](06-Records.md) | 实验记录规范（EX-XXX 模板） | ✅ |

## 快速定位

- **要跑实验** → [05-Execution-Plan.md](05-Execution-Plan.md)（步骤）+ [04-Work-Guide.md](04-Work-Guide.md)（命令）
- **要加新方法** → [03-Design-Spec.md](03-Design-Spec.md)（管线分层）+ [02-Technical-Standards.md](02-Technical-Standards.md)（规范）
- **要知道为什么这么设计** → `../report/04-synthesis-insights.md`
- **要知道当前做到哪** → `../report/00-README.md` + 根目录 `research-report/CONTINUE_GUIDE_NEW.md`

## 引用关系

```
docs/05 执行步骤 ──调用──> scripts/*.py（生成脚本）
        │                        │
        └──产出──> eval_data/{model}/singles/ ──评估──> eval/metrics_common.py + metrics_fundus.py
                                                        │
                                                        └─打分──> eval/score_scheme.py + plot_metrics.py
```

> 评估与打分脚本在**项目根目录** `eval/`，本目录的生成脚本产出对齐其输入口径（见 02 技术规范）。
