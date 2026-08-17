# 技术规范：数据口径 / 依赖 / CLI / 输出目录 / 命名

> 更新日期：2026-08-03 ｜ 对齐根目录 `docs/03-Development-Standards.md` 的风格

---

## 1. 数据口径（重要，评估体系依赖）

| 项 | 值 | 说明 |
|:--|:--|:--|
| 训练/素材数据 | `fundus/_all_images_ORIGINAL/`（322-330 张 JPG） | 真实严重 DR 图 |
| 输入分辨率 | 读入后 Resize 到 **128×128** | 与 `eval_data/real/` 和评估脚本 `load_images` 一致 |
| 值域 | PIL 读入 RGB uint8 [0,255]；生成后转回 0-255 存 PNG | 评估脚本内部转 [-1,1] |
| 输出分辨率 | 128×128 PNG | 严格等于评估 `--img_size 128` |
| 数量 | 每方法 **300 张** | 与现有深度方法一致，real 330 张 |

> ⚠️ 原图分辨率各异（1900~2572 方形），直接 Resize 即可（方形→方形无形变）。

## 2. 依赖

```
numpy
scikit-learn          # PCA / GMM（sklearn 1.9.0 已验证）
Pillow                # 图像 IO
opencv-python         # （可选）泊松融合 cv2.seamlessClone / Retinex 等
```

不引入深度学习框架（本目录不做深度方法）。评估阶段复用根目录 `eval/` 的 torch 依赖（已有）。

## 3. 统一 CLI 规范（每个生成脚本必须实现）

```bash
python scripts/<method>_gen.py \
    --data ../../fundus/_all_images_ORIGINAL \   # 素材/训练图
    --output ../../eval_data/<model> \            # 输出根目录
    --img_size 128 \                              # 分辨率（默认 128）
    --num_images 300 \                            # 生成数量
    --seed 42 \                                   # 随机种子
    [--<method>_param ...]                        # 方法特定参数
```

| 参数 | 类型 | 默认 | 说明 |
|:--|:--|:--|:--|
| `--data` | str | `../../fundus/_all_images_ORIGINAL` | 素材图目录（脚本以 `python scripts/xx.py` 在 `machine_learning/` 下执行，相对执行目录两级上跳到根目录） |
| `--output` | str | `../../eval_data/<model>` | 输出根目录（含 `singles/`） |
| `--img_size` | int | 128 | 生成分辨率 |
| `--num_images` | int | 300 | 生成数量 |
| `--seed` | int | 42 | 随机种子 |
| 方法特定 | — | — | 见各脚本 docstring |

**约定**：
- 脚本放 `scripts/`，文件名 `<method>_gen.py`
- 统一用 `argparse`，参数说明中文
- 脚本可直接运行（`if __name__ == "__main__"`），也可 import 复用内部函数
- 运行目录 = `generate_project/machine_learning/`（相对路径以此为准）

## 4. 输出目录结构

```
generate_project/machine_learning/
├── scripts/<method>_gen.py        # 生成脚本
├── report/                        # 文献报告（为什么）
├── docs/                          # 工程文档（怎么做，本文档所在）
├── <method>_config.json           # （可选）每次运行超参数存档
└── ../../eval_data/<model>/       # 输出（对齐评估体系）
    └── singles/
        ├── sample_0000.png
        ├── sample_0001.png
        └── ...（300 张）
```

**模型键名**（`<model>`，进 eval_data 和 MODEL_LABELS 的键）：
| 模型 | 键 | 显示名建议 |
|:--|:--|:--|
| PCA 线性重建 | `pca` | PCA 线性重建 |
| GMM 混合采样 | `gmm` | GMM 混合采样 |
| 补丁拼接 | `patch` | 补丁拼接 |
| 泊松病变重排 | `poisson` | 泊松病变重排 |
| Retinex 光照交换 | `retinex` | Retinex 光照交换 |

## 5. 命名与代码风格

- Python 4 空格缩进；类 PascalCase、函数/变量 snake_case、常量 UPPER_SNAKE_CASE
- 关键函数写 docstring（中英文均可）；argparse 说明中文
- 禁止 emoji
- 每个脚本头部 docstring 写明：方法原理、文献依据、用法示例、预期结果
- 每次运行建议把超参数打印到 stdout（复现留痕）

## 6. 已知坑

| 坑 | 说明 |
|:--|:--|
| Windows GBK 编码 | 所有文件读取/JSON 写用 `encoding="utf-8"`；控制台 `python -X utf8` |
| 中文路径 | 数据路径含中文（`fundus/_all_images_ORIGINAL`）无问题，但避免在输出路径加中文 |
| seed 一致性 | numpy/sklearn 各自设置 seed；sklearn 某些算法有 `random_state` 参数需显式传 |
| PCA 白化 | 用 `PCA(whiten=True)` 时 `inverse_transform` 会还原到原值域，需再 clip 0-255 |
