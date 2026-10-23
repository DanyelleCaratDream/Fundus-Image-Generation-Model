# pretrained/ —— 预训练模型方法

> 依据老师 Bug 2：「方法不限于深度学习，预训练/迁移学习都可以试」。

## 方法族

| 目录 | 方法 | 生成框架 | 状态 | 结论 |
|:--|:--|:--|:--|:--|
| [`SD21-LoRA/`](SD21-LoRA/) | Stable Diffusion 2.1 + 眼底 LoRA + 自训 severe LoRA（B1 线） | 扩散（潜空间） | ✅ 已完成 | 真实感好但病理弱，保留为对照 |
| [`TextGuidedDiffusion/`](TextGuidedDiffusion/) | 文本引导 + ControlNet（Tiger-Model 参考，罕见病理增广） | 扩散（潜空间） | 📌 代码已下载，未跑 | 待验证；作者要求后续提醒 |

每个方法族目录自含 `train/ infer/ tools/ data/ results/ judgment/` 与 README（方法标注、结果、运行方式）。

**归类原则**：这里的“方法族”按**初始化/先验来源**划分——凡是依赖预训练模型（SD / ControlNet）
再做迁移或适配的，都放本目录；从零自研训练的扩散放 `deep_learning/Fundus-Diffusion/`。
生成框架（扩散 / Flow Matching / GAN）是另一条正交的轴。

## 资料

- `report-pretrained-lit.md` —— 预训练/迁移/生成模型扩充的文献归档（Phase B 结论 + 2026-08-03 调研补）。
- `research-report/evaluation_report.md`（Phase B 调研）+ `docs/05-Research-Methodology.md`。
