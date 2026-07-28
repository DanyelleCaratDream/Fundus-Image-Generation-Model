# 开发规范

## 1. 目录结构规范

每个模型项目（Fundus-VAE、Fundus-GAN 等）内部结构：

```
Fundus-XXX/
├── STANDARDS.md          ← 本项目的技术规范（引用上级 docs/）
├── requirements.txt      ← 本项目特有的依赖
├── train.py              ← 统一训练入口
├── generate.py           ← 统一生成入口
├── model/
│   ├── __init__.py
│   ├── model.py          ← 模型定义
│   └── layers.py         ← 自定义层（可选）
├── utils/
│   ├── __init__.py
│   ├── dataset.py        ← 数据集定义
│   ├── env_check.py      ← 环境检查
│   └── visualize.py      ← 可视化工具
└── results/              ← 训练输出（由 --output_dir 指定）
```

## 2. 统一 CLI 接口规范

### train.py 必须实现：

```bash
python train.py \
    --epochs 800 \
    --batch_size 16 \
    --img_size 128 \
    --lr 0.0002 \
    --model_save_interval 50 \
    --image_save_interval 50 \
    --preview_grid_size 4 \
    --dataset_path "D:/fundus/_all_images_ORIGINAL" \
    --output_dir "./results" \
    --num_workers 4 \
    [--no_cuda] \
    [--resume "./results/models/checkpoint_epoch_200.pth"]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 200 | 总训练轮数 |
| `--batch_size` | int | 16 | 批次大小 |
| `--img_size` | int | 128 | 图像分辨率（2的幂次） |
| `--lr` | float | 0.0002 | 学习率 |
| `--model_save_interval` | int | 50 | 每N轮保存一次模型 |
| `--image_save_interval` | int | 50 | 每N轮保存一次生成图 |
| `--preview_grid_size` | int | 4 | 预览图网格 NxN |
| `--dataset_path` | str | 必填 | 数据集文件夹路径 |
| `--output_dir` | str | "./results" | 输出根目录 |
| `--num_workers` | int | 4 | DataLoader 线程数 |
| `--seed` | int | 42 | 随机种子 |
| `--no_cuda` | flag | false | 强制使用 CPU |
| `--resume` | str | None | 断点续训 checkpoint 路径 |

### generate.py 必须实现：

```bash
python generate.py \
    --checkpoint "./results/models/generator_epoch_800.pth" \
    --num_images 100 \
    --img_size 128 \
    --output_dir "./results/generated" \
    --grid_size 10 \
    [--no_cuda]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | 必填 | 模型权重路径 |
| `--num_images` | int | 100 | 生成图片数量 |
| `--img_size` | int | 128 | 分辨率 |
| `--output_dir` | str | "./generated" | 输出目录 |
| `--grid_size` | int | 10 | 网格 NxN (0=单张保存) |
| `--seed` | int | 42 | 随机种子 |
| `--no_cuda` | flag | false | 强制使用 CPU |

## 3. 输出目录结构

```
{output_dir}/
├── config.json            ← 训练配置（超参数存档）
├── images/                ← 训练中生成的预览图
│   ├── epoch_0001.png
│   ├── epoch_0050.png
│   └── ...
├── models/                ← 保存的模型权重
│   ├── generator_epoch_50.pth
│   ├── discriminator_epoch_50.pth
│   ├── optimizer_G_epoch_50.pth
│   └── ...
├── logs/                  ← 训练日志
│   ├── training.log
│   └── loss_curve.png
└── generated/             ← generate.py 输出
    ├── grid_100.png
    └── singles/
        ├── sample_0000.png
        └── ...
```

## 4. 代码风格
- Python 使用 4 空格缩进
- 类名使用 PascalCase，函数/变量使用 snake_case
- 常量使用 UPPER_SNAKE_CASE
- 禁止在代码中使用 emoji
- 关键函数必须写 docstring（中文或英文均可）
- argparse 参数说明使用中文（面向小白用户）

## 5. 断点续训
每个 epoch 保存时，除了模型权重还需保存优化器状态：
- `generator_epoch_N.pth` / `discriminator_epoch_N.pth`
- `optimizer_G_epoch_N.pth` / `optimizer_D_epoch_N.pth`

断点续训时从 resume 路径加载权重，从记录的 epoch 继续训练。
