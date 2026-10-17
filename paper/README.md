# paper/ — 论文投稿文本（唯一成稿区）

> 建立于 2026-10-17。存放论文正文草稿：路线 A《小样本眼底图像生成：失效机理与评估方法》先行。
> **写作宪法 = [docs/14-Paper-Spine.md](../docs/14-Paper-Spine.md)**（主张→证据映射、口径统一表 16 条、术语中英对照、写作纪律）。数据权威 = [docs/15-Data-Documentation.md](../docs/15-Data-Documentation.md)。文献定位 = [docs/12](../docs/12-Literature-Survey-and-Positioning.md)。

## 写作约定（每节动笔前读一遍）

1. **语言**：中文写作，关键术语首现随文标英文；新术语回写 docs/14 第四节（单一词表，不得自建词表）。
2. **数字不手写**：每个数字带源标签（`DL-EX-xxx` / `EX-xxx` / JSON 文件名）；放行前与 EXPERIMENTS.md、eval_data JSON 三处对上（三点头核验）。
3. **分数必标尺度**：任何 0-100 分，同句标注（人工/human，单人 N=6）或（六维门控/gated，`_scores.json` v1，11 模型现版）。两套口径永不混排；旧文档的 72.9/44.2 是历史版本分，禁用。
4. **占位语法**：`【占位:xxx】`，grep 友好；占位句只写"将在 X 节报告结果"，**禁止预写结论**。
5. **引用红线**（docs/12）：「通用指标看不出医学语义差异」只能当共识**新实例**且必引 Deo 2025（arXiv:2505.07175）；禁说「首创病灶级指标」（Asokan 2026 LFS 在，划界：实例级 vs 分布级、真实病灶参照 vs 条件 mask 参照）；血管 mask 条件生成 2017 已有（Guibas/Costa）；引 Asokan 按原文口径（他们从未提 FID）。
6. **展示图 caption**：必须注明干净样本筛选事实（film_l1lpips 干净样本实测仅 ~25-30%，展示图为手工挑选，不可暗示随机样本）。
7. **结构改动**：先改 [outline.md](outline.md)（单一真源），再动章节文件。

## 文件清单

| 文件 | 内容 | 状态 |
|:--|:--|:--|
| [outline.md](outline.md) | 全书节级大纲（编号｜主张｜证据｜状态｜占位） | ✅ 随写作更新 |
| [abstract.md](abstract.md) | 摘要/标题 | stub |
| [ch1-introduction.md](ch1-introduction.md) | 引言 | stub |
| [ch2-related-work.md](ch2-related-work.md) | 相关工作 | stub |
| [ch3-method.md](ch3-method.md) | 方法（今晚：3.3 损失设计 + 3.4 骨架） | 🏃 |
| [ch4-experimental-setup.md](ch4-experimental-setup.md) | 实验设置（今晚：4.1 数据卫生 + 4.3 半节） | 🏃 |
| [ch5-results.md](ch5-results.md) | 结果——失效机理链主战场 | 🏃 |
| [ch6-discussion.md](ch6-discussion.md) | 讨论（今晚：6.1/6.2） | 🏃 |
| [ch7-limitations-conclusion.md](ch7-limitations-conclusion.md) | 局限与结论 | stub |
| [refs-notes.md](refs-notes.md) | 引用便签（.bib 建库前暂存） | ✅ |
| [figures/](figures/) | 图规格约定 + 占位清单 | ✅ |

## 证据源速查

- 机理叙事：`research-report/REPORT_DL_text.txt`（REPORT_DL.docx 文本版，**停更于 2026-08-02**——D1–D4 与数据卫生不在此文件，须引 EXPERIMENTS.md / docs/15）
- 数字唯一来源：`generate_project/deep_learning/EXPERIMENTS.md`（DL-EX-xxx）、`generate_project/machine_learning/EXPERIMENTS.md`（EX-xxx）、`eval_data/*.json`、`eval/human_scores.csv`
