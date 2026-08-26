# 工作说明手册（操作手册）

> 常用命令 + 已知坑。新会话跑任务前先读本文件 + `research-report/CONTINUE_GUIDE_NEW.md`。
> 更新日期：2026-08-02

---

## 一、环境确认

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- 当前：PyTorch 2.11.0+cu130，RTX 4060 Laptop（实际 ~7GB 可用）
- 评估依赖（已装）：`pytorch-fid 0.3.0` `prdc 0.2` `pytorch-msssim 1.0.0` `piq 0.8.0`
- 项目依赖：torch/torchvision/cv2/lpips/scipy/matplotlib

## 二、数据集与条件图

- 训练基准集：`fundus/_all_images_ORIGINAL/`（330 张）
- 血管骨架条件图：`generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/`（330 张 128×128，条件扩散 + Vessel Dice 用）
- 重新生成骨架：`python prep_conditions.py --dataset_path "../../../../fundus/_all_images_ORIGINAL" --output_dir "./conditions" --mode mask --img_size 128`（在 `generate_project/deep_learning/Fundus-Diffusion/ddpm/` 下）

## 三、生成评估图（Diffusion / GAN / VAE）

> 所有生成脚本都在各自目录下运行（`from train import ...` 依赖 cwd）。**路径按各自项目目录相对根目录**：Diffusion/GAN 子项目 4 层 `../../../../`，VAE 3 层 `../../../`。
> generate.py 会自动建 `{output_dir}/singles/` 子目录存独立单张图 —— 评估 `--fake` 必须指向该子目录。
> **评估统一用 DDIM 50 步、seed 42**（Phase A 同配置）。
> **⚠️ 样本量新规（2026-08-18）**：重新生成统一 `--num_images 60`（不再 300 全量）；旧模型 300 张评估记录不变。

### Diffusion（`generate_project/deep_learning/Fundus-Diffusion/ddpm/`）
```bash
# ① 最佳模型 FiLM+L1+LPIPS（条件，base_dim=128）
python generate.py --checkpoint results_film_l1lpips/models/final_model.pth \
    --num_images 60 --output_dir ../../../../eval_data/film_l1lpips --grid_size 0 \
    --sampler ddim --sampling_steps 50 --cond_path ./conditions \
    --base_dim 128 --dim_mults 1 2 3 4 --attn_layers 2 --seed 42

# ② FiLM MSE（条件，base_dim=128）：同上，--checkpoint 换 results_film/models/final_model.pth
# ③ 条件扩散（条件，base_dim=64）：--checkpoint 换 results_cond/models/final_model.pth，--base_dim 64
# ④ 基础 DDPM（无条件，base_dim=64）：--checkpoint 换 "results_去掉ColorJitter版/models/final_model.pth"，去掉 --cond_path，--base_dim 64
```

### DCGAN（`generate_project/deep_learning/Fundus-GAN/dcgan/`）
```bash
python generate.py --checkpoint results_220726_020708/models/checkpoint_epoch_001500.pth \
    --num_images 60 --output_dir ../../../../eval_data/dcgan \
    --img_size 128 --latent_dim 100 --grid_size 0 --seed 42
```

### VAE（`generate_project/deep_learning/Fundus-VAE/`）
```bash
python generate.py --checkpoint results/vanilla_vae_large_210726_212102/models/final_model.pth \
    --num_images 60 --output_dir ../../../eval_data/vae \
    --img_size 128 --latent_dim 256 --dim 64 --grid_size 0 --seed 42
```

## 四、评估（在项目根目录运行）

### 通用层指标（FID/KID/MMD/IS/Precision-Recall/Density-Coverage/1-NN/MS-SSIM/LPIPS + 颜色统计）
```bash
python eval/metrics_common.py --real eval_data/real --fake eval_data/<model>/singles \
    --img_size 128 --device cuda --json
```
- 输出：`eval_data/<model>_metrics.json`
- `--json` 只输出 JSON（prdc 会污染 stdout，已内部抑制）

### 专用层自设计指标（病灶/血管/Vessel Dice/记忆检测/C2ST/BRISQUE）
```bash
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/<model>/singles \
    --model <model> --device cuda \
    [--cond_path generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions] [--skip_c2st] --json
```
- `--cond_path` 仅条件模型需要（血管骨架条件图）；`--skip_c2st` 可跳过 C2ST 的 CNN 训练加速
- 输出：`eval_data/<model>_fundus_metrics.json`

### 可视化（柱状图 + 雷达图，可复用）
```bash
python eval/plot_metrics.py                # 输出到 research-report/figures/
python eval/plot_metrics.py --outdir <dir> # 自定义输出目录
```
- 自动扫描 `eval_data/` 所有 `*_metrics.json` + `*_fundus_metrics.json`，生成四张图：general_metrics（通用层）/ color_metrics（颜色）/ fundus_metrics（专用层）/ radar（归一化雷达图）；若存在 `_scores.json` 再自动补第五张 score_overview（综合评分总览）
- **新增方法（如 Phase C2 传统 ML）只需把结果 JSON 放进 eval_data/ 后重跑**，新模型自动进图；显示名可在 `eval/plot_metrics.py` 的 `MODEL_LABELS` 补充

### 综合评分（人工分校准 + 0-100 总分，可复用）
> 为什么这么设计（权重理由/门控动机/指标去留判据/局限）→ 读 `docs/09-Score-Scheme-Design.md`
```bash
python eval/score_scheme.py                    # 校准表 + 六维门控评分表 → eval_data/_scores.json
python eval/score_scheme.py --no-gate          # 对照输出未门控"原始分"（VAE 会跳回 ~58 证明门控必要）
python eval/score_scheme.py --models a,b       # 只看子集
python eval/score_scheme.py --scorecard        # 额外画 scorecard.png（总分色带 + 六维雷达）到 figures/
```
- 六维加权（D1 病灶 .30 / D2 抗识破+分布 .25 / D3 多样性 .20 / D4 血管 .10 / D5 颜色 .08 / D6 记忆 .07）+ 现实主义门控 `R=D2`：`总分 = 100×[0.30D1R+0.25D2+0.20D3+0.10D4R+0.08D5R+0.07D6]`
- **新方法跑完评估 → 丢 JSON 进 eval_data/ → 重跑 score_scheme.py 即得客观总分**（无需人工分，仅缺人工分时不参与校准）；缺键自动容错（整维缺失权重重归一化）
- 校准表 = 每指标 Spearman ρ vs 人工分 + 组内最优是否==人工最优，判据写死在 `CANDIDATE_METRICS`（进计分/仅参考）
- 复用 `plot_metrics.py` 的模型元信息（`import plot_metrics`），新模型显示名在 `MODEL_LABELS` 补即可

## 五、git 推送（双远端）

```bash
git push github main && git push gitee main
```

> ⚠️ `eval_data/` 的图片（1800+ 张）已 .gitignore 排除，只提交根目录的 `*_metrics.json` 评估结果。`fundus/` 数据集同理不提交。

## 六、已知坑（重要）

### #1 `torch.load` 的 weights_only 报错（✅ 已修复 2026-08-01）
PyTorch 2.6+ 默认 `weights_only=True`，旧 checkpoint 会报 `Unsupported global`。
**已修复**：DCGAN/VAE 的 generate.py 已加 `weights_only=False`，DDPM generate.py 自带。无需再改。

### #2 条件扩散 / 基础 DDPM checkpoint size mismatch（✅ 已修复 2026-08-01）
旧版 UNet（cond/base_cj）加载时会报 `upblocks.*.temb_proj` size mismatch。
**已修复**：generate.py 内置自动检测（`block2.0.` 键）→ 禁用 FiLM + key 重映射（`block2.0→norm2`、`block2.3→conv2`），`base_dim=64` 加载。验证 missing=0/unexp=0。

### #3 控制台中文乱码
Windows GBK 控制台显示中文乱码是**显示问题**，不影响文件内容。写文件用 UTF-8；跑含中文输出的脚本可用 `python -X utf8 script.py` 规避。

### #4 评估图必须重新生成
历史 `results*/images/` 里全是 4×4 网格预览图（522×522），**不能**直接用于评估。必须用第三节命令批量生成独立单张图（`singles/`）。

### #5 数据增强颜色禁忌
眼底图的橙红调来自血红蛋白吸收，有生理学意义。**禁止重 ColorJitter**（项目已因 ColorJitter 污染参数踩坑）。增广以几何变换为主（如 `fundus/rotate_augment_check.py` 的旋转检查）。
