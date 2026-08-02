# 技术栈说明

## 基础环境
| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | 推荐 3.11 或 3.12 |
| PyTorch | 2.11.0+cu130 | GPU 版，CUDA 13.0 |
| CUDA Toolkit | 13.0 | 由 PyTorch 自带 |
| cuDNN | 已验证 | PyTorch 已确认 cuDNN 可用 |

## 核心依赖
| 包名 | 用途 | 安装确认 |
|------|------|---------|
| torch | 深度学习框架 | `python -c "import torch; print(torch.__version__)"` |
| torchvision | 图像处理工具集 | `python -c "import torchvision; print(torchvision.__version__)"` |
| numpy | 数值计算 | |
| Pillow | 图像 I/O | |
| matplotlib | 绘图（Loss 曲线等） | |
| scipy | FID 分数计算（矩阵平方根） | `pip install scipy` |
| cv2 | 血管骨架提取、经典图像处理 | `pip install opencv-python` |
| lpips | 感知损失 / 评估 | `pip install lpips` |
| pytorch-fid | FID / IS / KID | `pip install pytorch-fid` |
| prdc | Precision/Recall, Density/Coverage | `pip install prdc` |
| pytorch-msssim | MS-SSIM 多样性 | `pip install pytorch-msssim` |
| piq | BRISQUE/NIQE 无参考质量 | `pip install piq` |

## GPU 环境验证
每次训练前运行以下命令确认 GPU 就绪：

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

输出应类似：
```
PyTorch: 2.11.0+cu130
CUDA Available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8.0 GB
```

## 显存预算参考（RTX 4060 8GB）

| img_size | batch_size | 显存占用 |
|----------|-----------|---------|
| 64x64 | 32 | ~2-3 GB |
| 128x128 | 16 | ~4-5 GB |
| 128x128 | 32 | ~6-7 GB |
| 256x256 | 16 | ~7-8 GB (接近极限) |

## 代码规范
- 所有 `train.py` 使用 `argparse` 解析命令行参数
- 参数命名统一为 `--param_name` 格式
- 代码中不使用 emoji 字符
- 文件编码统一使用 UTF-8
