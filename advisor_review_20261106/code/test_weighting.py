#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""加权损失 + 同步增强 单元测试（CPU）

运行
    python tools/test_weighting.py
全部通过时打印 ALL TESTS PASSED，退出码 0。
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CFM = Path(__file__).resolve().parents[1]
DL = CFM.parents[1]
REPO = DL.parents[1]
sys.path.insert(0, str(DL / "Fundus-Diffusion" / "ddpm"))


def weighted_loss(pred, target, lw, lam, loss_type="l1"):
    w = (1.0 + lam * lw).expand_as(pred)
    err = (pred - target) ** 2 if loss_type == "mse" else (pred - target).abs()
    return (err * w).sum() / w.sum()


def test_lambda_zero_equals_plain_l1():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    lw = torch.zeros(2, 1, 16, 16)
    got = weighted_loss(pred, target, lw, 0.0)
    want = F.l1_loss(pred, target)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5), (got, want)
    print("PASS 1: lambda=0 equals F.l1_loss")


def test_zero_mask_independent_of_lambda():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    lw = torch.zeros(2, 1, 16, 16)
    got = weighted_loss(pred, target, lw, 8.0)
    want = F.l1_loss(pred, target)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)
    print("PASS 2: zero mask + lambda=8 equals F.l1_loss")


def test_matches_explicit_channel_repeat():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    lw = torch.rand(2, 1, 16, 16)
    w = 1.0 + 8.0 * lw
    ref = (torch.abs(pred - target) * w.repeat(1, 3, 1, 1)).sum() / w.repeat(1, 3, 1, 1).sum()
    got = weighted_loss(pred, target, lw, 8.0)
    assert torch.allclose(got, ref, atol=1e-6)
    print("PASS 3: expand_as matches explicit repeat")


def test_regression_old_expression_was_three_times():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    lw = torch.zeros(2, 1, 16, 16)
    w = 1.0 + 8.0 * lw
    old = (torch.abs(pred - target) * w).sum() / w.sum()
    assert torch.allclose(old, 3.0 * F.l1_loss(pred, target), atol=1e-6)
    print("PASS 4: old expression documented as 3x")


def test_mse_branch():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    lw = torch.zeros(2, 1, 16, 16)
    got = weighted_loss(pred, target, lw, 0.0, loss_type="mse")
    want = F.mse_loss(pred, target)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5)
    print("PASS 5: mse branch respected")


def test_synchronized_augmentation():
    """非对称合成图/权重：同一 flip+rotate 后仍对齐；只变换图像则错位。"""
    rng = np.random.RandomState(0)
    img = np.zeros((64, 64, 3), np.float32)
    img[8:20, 6:14] = 1.0                     # 左上角亮块（非对称）
    w = np.zeros((64, 64), np.float32)
    w[8:20, 6:14] = 1.0
    img_pil = TF.to_pil_image(img)
    do_flip, angle = True, 12.0
    img_t = TF.rotate(TF.hflip(img_pil), angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    w_t = TF.rotate(TF.hflip(torch.from_numpy(w)[None]), angle,
                    interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)[0].numpy()
    img_a = np.asarray(img_t, np.float32)
    # 对齐：亮块位置与权重位置重合
    sel_img = img_a.mean(-1) > 0.5
    sel_w = w_t > 0.5
    iou = (sel_img & sel_w).sum() / max(1, (sel_img | sel_w).sum())
    # 双线性插值使边缘略有差异，0.7 已足以区分"对齐"与"错位"
    assert iou > 0.7, f"synchronized aug misaligned, IoU={iou}"
    # 只翻转图像、权重不动：应明显错位
    img_f = np.asarray(TF.hflip(img_pil), np.float32)
    sel_img_f = img_f.mean(-1) > 0.5
    sel_w0 = w > 0.5
    iou_bad = (sel_img_f & sel_w0).sum() / max(1, (sel_img_f | sel_w0).sum())
    assert iou_bad < 0.05, f"expected misalignment demo, IoU={iou_bad}"
    print(f"PASS 6: synchronized aug IoU={iou:.3f}; flip-only IoU={iou_bad:.3f}")


def test_dataset_shapes_and_augment():
    from train import FundusDataset
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = FundusDataset(str(REPO / "fundus" / "combined_sq256"), transform=transform,
                       cond_root=str(CFM / "data" / "conditions_combined_256"),
                       img_size=128,
                       lesion_root=str(CFM / "data" / "lesion_weights_256"),
                       augment=True)
    torch.manual_seed(123)
    a = ds[0]
    torch.manual_seed(123)
    b = ds[0]
    for ta, tb in zip(a[:3], b[:3]):
        assert torch.allclose(ta, tb), "same seed must give identical augmented sample"
    assert a[0].shape == (3, 128, 128) and a[1].shape == (1, 128, 128) and a[2].shape == (1, 128, 128)
    print("PASS 7: dataset shapes + deterministic under same seed")


if __name__ == "__main__":
    test_lambda_zero_equals_plain_l1()
    test_zero_mask_independent_of_lambda()
    test_matches_explicit_channel_repeat()
    test_regression_old_expression_was_three_times()
    test_mse_branch()
    test_synchronized_augmentation()
    test_dataset_shapes_and_augment()
    print("ALL TESTS PASSED")
