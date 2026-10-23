#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFM 训练入口（薄封装）

真正实现：../../Fundus-Diffusion/ddpm/train.py --objective cfm（共用 FiLM UNet / 数据管线 /
EMA / 温度守护）。本脚本只负责三套预设配置与绝对路径，避免复制训练器造成代码分叉。

用法
    python train/train_cfm.py --preset cfm256_lesion
    python train/train_cfm.py --preset cfm256_lesion --epochs 1600   # 额外参数原样透传
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CFM = HERE.parents[1]                    # Fundus-FlowMatching/CFM
FLOW = CFM.parent                        # Fundus-FlowMatching
DL = FLOW.parent                         # deep_learning
REPO = DL.parents[1]                       # 仓库根（deep_learning → generate_project → 仓库根）
DDPM = DL / "Fundus-Diffusion" / "ddpm"

PRESETS = {
    # A 域最佳权重 → 老师 128 数据（CFM@128，320 轮）
    "cfm128": [
        "--objective", "cfm",
        "--dataset_path", str(REPO / "fundus/combined_sq128"),
        "--cond_path", str(DDPM / "conditions_combined_128"),
        "--resume", str(DDPM / "results_film_l1lpips/models/final_model.pth"),
        "--img_size", "128", "--base_dim", "128", "--batch_size", "2",
        "--epochs", "1100",
        "--output_dir", str(CFM / "results/cfm128"),
    ],
    # CFM@128 → 原生 256（320 轮）
    "cfm256": [
        "--objective", "cfm",
        "--dataset_path", str(REPO / "fundus/combined_sq256"),
        "--cond_path", str(CFM / "data/conditions_combined_256"),
        "--resume", str(CFM / "results/cfm128/models/final_model.pth"),
        "--img_size", "256", "--base_dim", "128", "--batch_size", "1",
        "--epochs", "1420",
        "--output_dir", str(CFM / "results/cfm256"),
    ],
    # CFM@256 → 病灶加权续训（150 轮）
    "cfm256_lesion": [
        "--objective", "cfm",
        "--dataset_path", str(REPO / "fundus/combined_sq256"),
        "--cond_path", str(CFM / "data/conditions_combined_256"),
        "--lesion_weight_dir", str(CFM / "data/lesion_weights_256"),
        "--lesion_lambda", "8",
        "--resume", str(CFM / "results/cfm256/models/final_model.pth"),
        "--img_size", "256", "--base_dim", "128", "--batch_size", "1",
        "--epochs", "1570",
        "--output_dir", str(CFM / "results/cfm256_lesion"),
    ],
}

COMMON = [
    "--loss_type", "l1", "--use_lpips", "--lpips_weight", "0.1",
    "--color_weight", "0.01", "--cool_sleep", "-0.5",
    "--num_workers", "0", "--seed", "42", "--image_save_interval", "25",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()
    cmd = [sys.executable, str(DDPM / "train.py")] + PRESETS[a.preset] + COMMON + extra
    print(" ".join(cmd), flush=True)
    if a.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=str(DDPM)))


if __name__ == "__main__":
    main()
