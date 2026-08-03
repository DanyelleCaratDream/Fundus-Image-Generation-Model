# 宏观设计规范：generate_project 四方向分工

> 更新日期：2026-08-03 ｜ 本文件讲"四方向怎么分工、共用什么基础设施"；各方向内部的算法/管线设计见各自 docs。

---

## 1. 四方向定位（回答老师 Bug 2：方法不限于深度学习）

| 方向 | 定位 | 状态 | 负责人文档 |
|:--|:--|:--|:--|
| **deep_learning/** | 深度学习从零训练：Diffusion（主力）/ GAN / VAE | ✅ 已完成评估 | 各项目 STANDARDS.md |
| **machine_learning/** | 传统 ML / 经典 CV（非深度）生成 | 📌 **当前**（Phase C2） | `machine_learning/docs/03-Design-Spec.md` |
| **pretrained/** | 基于预训练模型生成（SD LoRA 等） | 📌 Phase C3 可选 | `pretrained/` |
| **transfer_learning/** | 迁移学习（权重初始化/域适配/特征复用） | 📌 Phase C | `transfer_learning/` |

**分工原则**：同一个生成问题，按"是否训练 / 是否预训练 / 是否迁移"分四条技术路线探索，最后统一评估对比。

## 2. 共用基础设施（不得重复造）

所有方向的生成结果必须对齐同一套评估口径：

| 项 | 规范 | 来源 |
|:--|:--|:--|
| 图像口径 | 128×128 RGB，输出 `eval_data/{model}/singles/sample_XXXX.png` | `eval/metrics_common.py` 的 `load_images` |
| 每模型数量 | 300 张评估图（real 330 张） | Phase A 确立 |
| 通用层指标 | FID/KID/MMD/IS/P&R/Density/Coverage/1-NN/MS-SSIM/LPIPS + 颜色 | `eval/metrics_common.py` |
| 专用层指标 | 病灶/血管/记忆/C2ST/BRISQUE | `eval/metrics_fundus.py` |
| 综合评分 | 六维门控 0-100（D1 病灶/D2 抗识破/D3 多样/D4 血管/D5 颜色/D6 记忆） | `eval/score_scheme.py` |
| 可视化 | 自动扫描 eval_data JSON 出图 | `eval/plot_metrics.py` |

**接入协议**（任何新方向的方法）：
```
生成图 → eval_data/{model}/singles/ → 
  python eval/metrics_common.py --real eval_data/real --fake eval_data/{model}/singles --img_size 128 --device cuda --json →
  python eval/metrics_fundus.py ... → 
  python eval/score_scheme.py --scorecard && python eval/plot_metrics.py
```
模型显示名在 `eval/plot_metrics.py` 的 `MODEL_LABELS` 补一行。

## 3. 数据与真实图约束

- 训练/素材数据：`fundus/_all_images_ORIGINAL/`（330 张严重 DR，KW IV 级）
- 血管骨架 mask（条件扩散用）：`generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/`
- **记忆风险**：330 张处于小样本高风险区，任何方向的方法必须做"复制检测"（见 `machine_learning/docs/03` §5 三件套）

## 4. 统一 CLI / 命名 / 代码风格

- 对齐根目录 `docs/03-Development-Standards.md`（argparse 中文、seed、输出结构）
- 生成脚本统一 `<method>_gen.py` 命名（ML 方向）或各项目自带 train/generate.py（深度方向）
- Python 4 空格、docstring 中英文均可、禁止 emoji
- Windows 下文件读写一律 `encoding="utf-8"`，控制台 `python -X utf8`

## 5. 设计决策链（为什么这样分工）

1. 老师 Bug 2 要求"方法不限于深度学习" → 需要明确的非深度探索方向
2. 深度从零训练已做完（Diffusion 胜出）→ 作为对照基线，不重做
3. 传统 ML 是"当前优先"：成本低、有眼底先例（Fiorini/Magnusson）、能出"新病灶组合"
4. 预训练/迁移学习放到 C3/后续：文献显示医学预训练不必然更好，价值待验证
5. 所有方向最终用**同一把尺子**（六维评分 + 下游分类器 TSTR/TRTR）验收
