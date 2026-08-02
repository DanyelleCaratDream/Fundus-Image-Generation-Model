# eval/ 两层指标脚本

Phase A 评估体系的两个脚本（对应报告 `research-report/evaluation_report.md` 第一部分）。

- `metrics_common.py` —— **通用层**：FID / KID / MMD / IS / Precision-Recall / Density-Coverage / 1-NN / MS-SSIM / LPIPS + 颜色统计（RGB 均值/方差/直方图距离）
- `metrics_fundus.py` —— **专用层自设计**：病灶（出血/渗出 Wasserstein 距离 + 保留率）、血管 Wasserstein / Vessel Dice、记忆检测（NN-SSIM + 复制率）、C2ST 真伪分类、BRISQUE

## 运行（在项目根目录）

```bash
# 通用层
python eval/metrics_common.py --real eval_data/real --fake eval_data/<model>/singles \
    --img_size 128 --device cuda --json

# 专用层（条件模型加 --cond_path；--skip_c2st 可跳过 C2ST 训练加速）
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/<model>/singles \
    --model <model> --device cuda \
    [--cond_path generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions] [--skip_c2st] --json
```

- `--fake` 指向生成图的 `singles/` 子目录（由 generate.py 自动产出）
- 输出：`eval_data/<model>_metrics.json` 和 `eval_data/<model>_fundus_metrics.json`
- 依赖：`pytorch-fid` `prdc` `pytorch-msssim` `piq` `lpips`；FID 的 Inception 权重需首次联网下载

## 已知注意事项

- FID 为小样本有偏估计，比较时保证两侧样本量一致（本报告固定 real 330 vs 生成 300），且只做同量横向对比
- MS-SSIM 需要 ≥160px 输入，脚本内部会自动 resize 256 再算
- `prdc` 会污染 stdout，`--json` 模式下已内部用 `redirect_stdout` 抑制
