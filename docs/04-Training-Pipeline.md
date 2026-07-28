# 训练管线设计

## 统一训练流程

每个模型的训练管线遵循以下流程：

```
数据集加载 → 环境检查 → 模型初始化 → 训练循环 → 保存结果
```

### 1. 数据集加载
- 从 `--dataset_path` 扫描所有常见格式图片（jpg/png/bmp/tiff）
- 预处理：Resize → CenterCrop → ToTensor → Normalize([0.5]*3, [0.5]*3)
- 可选预加载到内存（--no_preload 禁用）
- DataLoader 配置：shuffle=True, drop_last=True, pin_memory=True

### 2. 环境检查
- 检查 CUDA 可用性
- 输出 GPU 型号和显存
- 输出 PyTorch 版本和 CUDA 版本

### 3. 训练循环
每个 epoch:
1. 遍历 DataLoader 中的所有 batch
2. 前向传播计算损失
3. 反向传播更新参数
4. 记录 loss 到日志文件
5. 定期保存模型 + 生成预览图 + 更新 Loss 曲线

### 4. 安全中断
- 监听 Ctrl+C（KeyboardInterrupt）
- 中断时保存当前模型权重，避免前功尽弃

## 各模型训练管线差异

### VAE
| 步骤 | 详情 |
|------|------|
| 前向 | Encoder → mu/logvar → reparameterize → Decoder → recon |
| 损失 | MSE (reconstruction) + beta * KL divergence |
| 优化器 | Adam |
| 生成 | 直接从高斯噪声采样 → Decoder 生成 |

### DCGAN
| 步骤 | 详情 |
|------|------|
| 前向 | G(z) → fake_img, D(real) + D(fake) → validity |
| 损失 | BCELoss (Generator: minimize log(1-D(G(z))), D: classify real/fake) |
| 优化器 | Adam (lr=0.0002, betas=(0.5, 0.999)) |
| 稳定技巧 | label smoothing, instance noise, R1 penalty |

### WGAN-GP
| 步骤 | 详情 |
|------|------|
| 前向 | G(z) → fake_img, D critic scores |
| 损失 | Wasserstein distance + gradient penalty |
| 优化器 | Adam (lr=0.0001, betas=(0.0, 0.9)) |
| 特点 | n_critic=5 (D训5次 G训1次) |

### StyleGAN2-ADA
| 步骤 | 详情 |
|------|------|
| 前向 | Mapping network → Synthesis network → Style modulation |
| 损失 | Logistic loss + R1 penalty + path length regularization |
| 优化器 | Adam (调参较复杂) |
| 特点 | ADA 自适应增强，适合小数据集 |

### DDPM
| 步骤 | 详情 |
|------|------|
| 前向（训练） | 加噪 x0 → xt, 预测噪声 epsilon |
| 损失 | MSE(pred_noise, true_noise) |
| 采样 | 从纯噪声逐步去噪 xT → x0 |
| 特点 | 训练稳定，采样慢（需1000步） |

### DDIM
| 步骤 | 详情 |
|------|------|
| 训练 | 同 DDPM |
| 采样 | 确定性采样，可减少步数（50-100步） |
| 特点 | 比 DDPM 快 10-50 倍 |
