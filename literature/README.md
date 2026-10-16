# 文献库

> 建立于 2026-10-15，用于集中存放**外部输入的文献资料**（老师给的、自己找的）。

## 放什么

| 类型 | 放哪 |
|---|---|
| 论文 PDF、综述 PDF | `pdf/` |
| 老师给的链接、arXiv 号、DOI | 写进 `links.md` |
| 你自己读完写的**调研笔记** | 不放这里——放回对应方向的 `report/`（见下） |

## 为什么 PDF 和笔记分开

- **PDF** 是原始素材，不用改，堆着就行。`.gitignore` 已排除 `*.pdf`，所以不进版本库（避免仓库膨胀）。
- **笔记**是你消化后的产物，要跟着主题走。项目里已有的文献笔记在：
  - `generate_project/machine_learning/report/01-literature-trad-ml.md`（传统 ML 生成方法）
  - `generate_project/machine_learning/report/02-literature-fundus.md`（眼底特异生成）
  - `generate_project/machine_learning/report/03-literature-augment.md`（小样本医学增广）
  - `generate_project/machine_learning/report/07-pretrained-gen-models.md`（预训练生成模型）
  - `generate_project/pretrained/report-pretrained-lit.md`、`generate_project/transfer_learning/report-tl-lit.md`

## 命名建议

`pdf/` 下的文件尽量用「作者年份-简短标题.pdf」，例如 `Zhang2022-RetinexLowLight.pdf`。
好认、好引用，写论文时直接对得上参考文献条目。

## 注意

`.gitignore` 排除了 `*.pdf`，所以**这些 PDF 不会被 git 跟踪**。
换电脑或重装时需要单独拷贝，不能靠 `git clone` 还原。
