#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""渗出加权损失诊断（CPU）：形状 / 3× 尺度 / 增强错位

2026-11-06 导师复核后新增。回答四个问题：
1. Dataset 返回的权重图是什么形状/范围？batch 后呢？
2. 旧加权公式在 λ=0 时是否等于 F.l1_loss？（预期：3×，即 bug）
3. 图像随机翻转/旋转后，权重图是否跟随？（预期：旧实现不跟随 → 错位）
4. film_l1lpips（CFM 的源头初始化）是否使用 cond_path？（决定其权重带不带错位历史）

用法
    python tools/diagnose_weighting.py
输出
    CFM/judgment/weight_misalign_demo.png
    eval_data/_audit/weighting_diagnosis.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

CFM = Path(__file__).resolve().parents[1]          # CFM/
DL = CFM.parents[1]                                 # deep_learning/
REPO = DL.parents[1]
sys.path.insert(0, str(DL / "Fundus-Diffusion" / "ddpm"))
from train import FundusDataset  # noqa: E402


def old_weighted_loss(pred, target, lw, lam):
    w = 1.0 + lam * lw
    return (torch.abs(pred - target) * w).sum() / w.sum()


def new_weighted_loss(pred, target, lw, lam, loss_type="l1"):
    w = (1.0 + lam * lw).expand_as(pred)
    err = (pred - target) ** 2 if loss_type == "mse" else (pred - target).abs()
    return (err * w).sum() / w.sum()


def exudate_mask(rgb01):
    """与 make_lesion_weights.py 相同的渗出阈值（仅用于对齐诊断）。"""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (r > 0.55) & (g > 0.38) & (b < 0.45) & ((r - b) > 0.15)


def main():
    out_json = REPO / "eval_data" / "_audit" / "weighting_diagnosis.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png = CFM / "judgment" / "weight_misalign_demo.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = FundusDataset(
        str(REPO / "fundus" / "combined_sq256"), transform=transform,
        cond_root=str(CFM / "data" / "conditions_combined_256"),
        img_size=256,
        lesion_root=str(CFM / "data" / "lesion_weights_256"),
        augment=True,
    )
    img, cond, w, idx = ds[0]
    report = {
        "sample_shapes": {"img": list(img.shape), "cond": list(cond.shape), "w": list(w.shape)},
        "dtypes": {"img": str(img.dtype), "cond": str(cond.dtype), "w": str(w.dtype)},
        "ranges": {"img": [float(img.min()), float(img.max())],
                   "cond": [float(cond.min()), float(cond.max())],
                   "w": [float(w.min()), float(w.max())]},
        "num_workers_note": "batch 后 w=[B,1,H,W]，predicted=[B,3,H,W]",
        "weight_files": len(list((CFM / "data" / "lesion_weights_256").glob("*.png"))),
    }

    # ---- 2) 损失尺度 ----
    torch.manual_seed(0)
    pred = torch.randn(1, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    zero_w = torch.zeros(1, 1, 64, 64)
    rand_w = torch.rand(1, 1, 64, 64)
    l1 = F.l1_loss(pred, target).item()
    old0 = old_weighted_loss(pred, target, zero_w, 0.0).item()
    old8 = old_weighted_loss(pred, target, zero_w, 8.0).item()
    new0 = new_weighted_loss(pred, target, zero_w, 0.0).item()
    new8 = new_weighted_loss(pred, target, zero_w, 8.0).item()
    old_r = old_weighted_loss(pred, target, rand_w, 8.0).item()
    new_r = new_weighted_loss(pred, target, rand_w, 8.0).item()
    report["loss_scale"] = {
        "plain_l1": l1,
        "old_lambda0": old0, "old_lambda0_over_plain": old0 / l1,
        "old_zero_mask_lambda8": old8, "old_zero_mask_lambda8_over_plain": old8 / l1,
        "new_lambda0": new0, "new_lambda0_over_plain": new0 / l1,
        "new_zero_mask_lambda8": new8, "new_zero_mask_lambda8_over_plain": new8 / l1,
        "old_random_mask_over_new": old_r / new_r,
    }
    print("[loss] old(l=0)/plain =", round(old0 / l1, 6),
          "| new(l=0)/plain =", round(new0 / l1, 6),
          "| old_random/new =", round(old_r / new_r, 6))

    # ---- 3) 增强错位演示 ----
    stem = Path(ds.image_paths[0]).stem
    img_pil = Image.open(ds.image_paths[0]).convert("RGB").resize((256, 256), Image.LANCZOS)
    wp = CFM / "data" / "lesion_weights_256" / f"{stem}.png"
    w_arr = np.asarray(Image.open(wp).convert("L").resize((256, 256), Image.LANCZOS), np.float32) / 255.0
    rgb = np.asarray(img_pil, np.float32) / 255.0
    mask = exudate_mask(rgb)
    def overlap(wmap, m):
        sel = wmap > 0.1
        if sel.sum() == 0:
            return None
        return float((sel & m).sum() / sel.sum())
    ov_before = overlap(w_arr, mask)
    flipped_img = np.asarray(TF.hflip(img_pil), np.float32) / 255.0
    ov_after = overlap(w_arr, exudate_mask(flipped_img))

    # 叠加图：权重=红，渗出阈值=绿
    def overlay(rgb_arr, wmap):
        o = (rgb_arr * 255).astype(np.uint8).copy()
        o[wmap > 0.1] = (0.5 * o[wmap > 0.1] + np.array([127, 0, 0])).astype(np.uint8)
        m = exudate_mask(rgb_arr)
        o[m] = (0.5 * o[m] + np.array([0, 127, 0])).astype(np.uint8)
        return Image.fromarray(o)
    panels = [
        ("image", img_pil),
        ("image + weight(red) + exudate mask(green)", overlay(rgb, w_arr)),
        ("image flipped (weight NOT flipped)", Image.fromarray((flipped_img * 255).astype(np.uint8))),
        ("flipped image + original weight (misaligned)", overlay(flipped_img, w_arr)),
    ]
    S, CAP = 256, 24
    canvas = Image.new("RGB", (4 * (S + 8) + 8, S + CAP + 16), "white")
    d = ImageDraw.Draw(canvas)
    for i, (lab, im) in enumerate(panels):
        x = 8 + i * (S + 8)
        canvas.paste(im.resize((S, S), Image.LANCZOS), (x, 8))
        d.text((x, S + 10), lab[:52], fill="black")
    canvas.save(out_png)
    report["misalignment"] = {
        "sample": stem,
        "weight_overlap_with_exudate_before_flip": ov_before,
        "weight_overlap_with_exudate_after_image_flip_only": ov_after,
        "demo_png": str(out_png),
    }
    print("[misalign] overlap before =", ov_before, "| after flip-only =", ov_after)

    # ---- 4) film_l1lpips 是否用 cond ----
    film_cfgs = []
    for cfg in list((DL / "Fundus-Diffusion" / "ddpm").glob("**/config.json")) + \
               list((CFM / "results").glob("*/config.json")):
        try:
            d0 = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "film_l1lpips" in str(cfg) or "film_l1lpips" in str(d0.get("output_dir", "")):
            film_cfgs.append({"config": str(cfg), "cond_path": d0.get("cond_path"),
                              "dataset_path": d0.get("dataset_path")})
    report["film_l1lpips_configs"] = film_cfgs

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", out_json)


if __name__ == "__main__":
    main()
