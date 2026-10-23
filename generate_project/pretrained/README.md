# pretrained/ —— 预训练模型方法

> 依据老师 Bug 2：「方法不限于深度学习，预训练/迁移学习都可以试」。

## 方法族

| 目录 | 方法 | 状态 | 结论 |
|:--|:--|:--|:--|
| [`SD21-LoRA/`](SD21-LoRA/) | Stable Diffusion 2.1 + 眼底 LoRA + 自训 severe LoRA（B1 线） | ✅ 已完成 | 真实感好但病理弱，保留为对照 |

每个方法族目录自含 `train/ infer/ tools/ data/ results/ judgment/` 与 README（方法标注、结果、运行方式）。

## 资料

- `report-pretrained-lit.md` —— 预训练/迁移/生成模型扩充的文献归档（Phase B 结论 + 2026-08-03 调研补）。
- `research-report/evaluation_report.md`（Phase B 调研）+ `docs/05-Research-Methodology.md`。
