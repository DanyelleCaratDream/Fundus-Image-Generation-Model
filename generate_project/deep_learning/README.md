# deep_learning/ —— 深度学习从零训练（已完成）

> Phase A/B 全部完成：6 个深度模型已训练并评估。本目录是历史成果存档 + 复现入口。
> 更新日期：2026-08-03

---

## 方法族总览

| 子项目 | 方法 | 状态 | 评估结果 |
|:--|:--|:--:|:--|
| [Fundus-Diffusion/](Fundus-Diffusion/) | DDPM 系列（无条件/条件/FiLM） | ✅ 最佳 | FiLM+L1+LPIPS **72.9 分**（最佳） |
| [Fundus-GAN/](Fundus-GAN/) | DCGAN / StyleGAN2 / WGAN-GP | ✅ DCGAN 已评，其余权重丢失跳过 | DCGAN 7.0 分（全面失败） |
| [Fundus-VAE/](Fundus-VAE/) | Vanilla VAE / Beta-VAE | ✅ | VAE Large 25.7 分（门控后） |

> 综合评分（六维门控 0-100）：film_l1lpips 72.9 / base_cj 44.2 / cond 44.0 / film 44.0 / vae 25.7 / dcgan 7.0。

## 各项目规范

- 每个子项目有 `STANDARDS.md`（技术规范：CLI/参数/输出结构）
- 训练入口 `train.py`、生成入口 `generate.py`（对齐根目录 `docs/03-Development-Standards.md`）
- 关键 checkpoint：
  - 最佳模型：`Fundus-Diffusion/ddpm/results_film_l1lpips/models/final_model.pth`
  - 血管骨架 mask（条件生成/评估用）：`Fundus-Diffusion/ddpm/conditions/`（330 张）

## 复现命令（评估图生成）

```bash
cd generate_project/deep_learning/Fundus-Diffusion/ddpm
python generate.py --checkpoint <ckpt> --num_images 300 --output_dir ../../../eval_data/<model> \
    --grid_size 0 --sampler ddim --sampling_steps 50 \
    --base_dim <64|128> --dim_mults 1 2 3 4 --attn_layers 2 \
    [--cond_path ./conditions] --seed 42
```

完整命令与评估：根目录 `docs/08-Work-Guide.md` + `research-report/evaluation_report.md`。

## 边界

本目录已完成，**不再新增训练**。后续新方法探索转到 `machine_learning/`（当前）、`pretrained/`、`transfer_learning/`。综合评估体系见根目录 `eval/`。
