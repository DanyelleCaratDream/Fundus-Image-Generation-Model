# deep_learning/ —— 深度学习从零训练

> Phase A/B 已全部完成（6 个深度模型训练并评估）。2026-10 起转向 **Phase D**：
> 在老师提供的 DR 分级数据集上训练生成器，产出可替换其 severe 类翻转副本的合成图。
> 更新日期：2026-10-16

---

## 📋 实验登记表

**[EXPERIMENTS.md](EXPERIMENTS.md)** —— 所有深度学习实验的编号、配置、结果与结论。
**写报告/论文时从这里取数**，不要另处手写。

> ⚠️ 编号用 **`DL-EX-xxx`** 前缀，与传统 ML 侧的 `EX-xxx`（`machine_learning/docs/06-Records.md`）区分——
> 两边原本都用 `EX-xxx` 且已撞车。

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
python generate.py --checkpoint <ckpt> --num_images 60 --output_dir ../../../../eval_data/<model> \
    --grid_size 0 --sampler ddim --sampling_steps 50 \
    --base_dim <64|128> --dim_mults 1 2 3 4 --attn_layers 2 \
    [--cond_path ./conditions] --seed 42
# ⚠️ 样本量新规 2026-08-18：统一 --num_images 60（不再 300 全量）；旧模型 300 张记录不变
```

完整命令与评估：根目录 `docs/08-Work-Guide.md` + `research-report/evaluation_report.md`。

## 边界

Phase A/B（6 模型训练评估）已完成。**当前唯一在跑的新训练是 Phase D 系列**（`Fundus-Diffusion/ddpm/` 下 D1/D2 及后续，见 EXPERIMENTS.md DL-EX-019/020）。方法探索的文献调研在 `machine_learning/`、`pretrained/`、`transfer_learning/`；综合评估体系见根目录 `eval/`。
