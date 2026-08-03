# 工作说明（操作手册）：传统 ML 生成眼底图

> 更新日期：2026-08-03 ｜ 本目录的常用命令 + 已知坑。评估体系命令见根目录 `docs/08-Work-Guide.md`。

---

## 一、环境确认

```bash
# 在项目根目录
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import sklearn; print(sklearn.__version__)"    # 传统 ML 用，已装 1.9.0
python -c "import cv2; print(cv2.__version__)"            # 泊松融合/Retinex 可选
```

## 二、生成评估图（在 `generate_project/machine_learning/` 下执行）

```bash
# 3 个确定性基线
python scripts/pca_gen.py    --num_images 300 --seed 42    # → ../../eval_data/pca
python scripts/gmm_gen.py    --num_images 300 --seed 42    # → ../../eval_data/gmm
python scripts/patch_gen.py  --num_images 300 --seed 42    # → ../../eval_data/patch

# 高价值组合（待建）
python scripts/poisson_gen.py  --num_images 300 --seed 42  # → ../../eval_data/poisson
python scripts/retinex_gen.py  --num_images 300 --seed 42  # → ../../eval_data/retinex

# 冒烟测试（先跑 5 张确认管线）
python scripts/pca_gen.py --num_images 5
```

**路径说明**：脚本在 `scripts/`，相对 `scripts/` 到根目录素材是 `../../fundus/...`，到 `eval_data` 是 `../../eval_data/...`。

## 三、评估（在项目根目录执行）

```bash
# 通用层（每模型一个）
python eval/metrics_common.py --real eval_data/real --fake eval_data/<m>/singles --img_size 128 --device cuda --json

# 专用层（条件模型加 --cond_path；--skip_c2st 可跳过 CNN 训练加速；传统 ML 无条件图跳过 vessel_dice）
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/<m>/singles --model <m> --device cuda --skip_c2st --json

# 打分 + 画图（自动入总分表/雷达图/综合评分总览）
python eval/score_scheme.py --scorecard
python eval/plot_metrics.py
```

## 四、复制检测三件套（质检，生成后必跑）

```bash
python scripts/memory_check.py --real ../../fundus/_all_images_ORIGINAL --fake ../../eval_data/<m>/singles
# 输出：NN-SSIM/LPIPS 距离分布 + 全图复制率 + 补丁重复率
```

## 五、新增模型显示名（进图必改）

`eval/plot_metrics.py` 的 `MODEL_LABELS`（及可选 `MODEL_COLORS`）补一行，否则图里显示裸键名：

```python
MODEL_LABELS = {
    ...,
    "pca": "PCA 线性重建",
    "gmm": "GMM 混合采样",
    "patch": "补丁拼接",
    "poisson": "泊松病变重排",
    "retinex": "Retinex 光照交换",
}
```

## 六、git 提交

```bash
cd d:/AI_Model_Project_for_Fundus_Color_Images
git add generate_project/machine_learning/ generate_project/pretrained/
git commit -m "docs+refactor: machine_learning 文档体系（报告+工程规范）+ pretrained 文献归档"
git push github main && git push gitee main
```

## 七、已知坑

| 坑 | 说明 |
|:--|:--|
| Windows GBK | 所有文件读取/JSON 写用 `encoding="utf-8"`；控制台 `python -X utf8` |
| 冒烟先于全量 | 任何新脚本先 `--num_images 5` 验证，再跑 300（避免写错路径浪费几分钟） |
| 评估脚本较慢 | metrics_fundus 的 C2ST 会训小 CNN，`--skip_c2st` 可先跳过加速 |
| 显示名漏改 | 新模型图里显示裸键名 → 记得改 MODEL_LABELS |
| 记忆检测维度 | score_scheme 有 D6 记忆维度，复制率高的方法会在这里被惩罚（符合设计） |
