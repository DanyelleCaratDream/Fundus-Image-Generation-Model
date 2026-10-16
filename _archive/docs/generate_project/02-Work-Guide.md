# 宏观工作说明（操作手册）：跨方向命令速查

> 更新日期：2026-08-03 ｜ 本文件是"跨方向快速入口"；单方向的详细命令见各子目录 Work-Guide。
> 完整操作手册：根目录 `docs/08-Work-Guide.md`。

---

## 一、环境确认

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # GPU: RTX 4060 8GB
python -c "import sklearn; print(sklearn.__version__)"  # 传统 ML 用
```

## 二、各方向入口速查

| 要做的事 | 去哪 | 命令示例 |
|:--|:--|:--|
| 深度模型生成评估图 | `generate_project/deep_learning/Fundus-*/` | 见各项目 STANDARDS.md |
| **传统 ML 生成（当前重点）** | `generate_project/machine_learning/` | `python scripts/pca_gen.py --num_images 60` |
| 预训练探索（C3） | `generate_project/pretrained/` | 尚未开始（先读文献） |
| 迁移学习（C） | `generate_project/transfer_learning/` | 尚未开始（先读文献） |

## 三、通用评估流程（全方向一致，在根目录执行）

```bash
cd d:/AI_Model_Project_for_Fundus_Color_Images

# 1) 通用层评估
python eval/metrics_common.py --real eval_data/real --fake eval_data/<model>/singles \
    --img_size 128 --device cuda --json

# 2) 专用层评估（--skip_c2st 可跳过 CNN 训练）
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/<model>/singles \
    --model <model> --device cuda [--skip_c2st] --json

# 3) 打分 + 画图
python eval/score_scheme.py --scorecard
python eval/plot_metrics.py
```

## 四、新方法接入 checklist（任何方向）

- [ ] 生成 60 张到 `eval_data/{model}/singles/`（128×128，`sample_0000.png` 起；2026-08-18 新规，不再 300）
- [ ] 通用层 + 专用层评估 JSON 产出
- [ ] `eval/plot_metrics.py` 的 `MODEL_LABELS` 补显示名
- [ ] score_scheme + plot_metrics 出分入图
- [ ] 复制检测（记忆风险，ML 方向见其 docs）
- [ ] 记录实验（各方向 `docs/06-Records.md`）

## 五、git 提交（双远端）

```bash
git add -A
git commit -m "<按类型写：docs:/feat:/refactor: ...>"
git push github main && git push gitee main
```

## 六、已知坑（跨方向）

| 坑 | 说明 |
|:--|:--|
| Windows GBK | 文件读写 `encoding="utf-8"`；控制台 `python -X utf8` |
| 显存 8GB | 评估脚本串行跑，不并行多个；必要时 `--skip_c2st` |
| 路径层级 | 生成脚本在各自项目目录下执行：一层项目（如 `machine_learning/`）到 `eval_data` 是 `../../eval_data`；`deep_learning/*/` 子目录（如 `ddpm/`）到根目录是 4 层 `../../../../` |
| 显示名 | 新模型忘了补 MODEL_LABELS → 图里显示裸键名 |
| 评估口径 | 必须 128×128、新方法 60 张（旧方法 300 张记录不变）、命名 `sample_XXXX.png`，否则评估脚本不识别 |
