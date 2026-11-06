#!/usr/bin/env python3
"""训练下游病灶分割模型（UNet++ / ResNet-18），仅在 RETOUCH Spectralis 子集上。

论文设定
--------
`For fluid lesion segmentation, we employ a UNet++ architecture [28] featuring a ResNet-18 [7]
encoder. To accurately gauge the quality of the reconstruction algorithms, this downstream model
is trained solely on the RETOUCH Spectralis subset.`

评估重建质量时，该模型对"重建后的低成本设备图像"做预测，与真值比较得到 DSC。因此本脚本
产出的模型会被 `eval_dice.py` 直接复用——**训练与评测必须使用完全相同的模型与预处理**，
否则 DSC 的差异无法归因到重建方法本身。

用法
----
    # 冒烟测试（2 个体积、2 个 epoch）
    python train_seg.py --csv retouch_spectralis.csv --out-dir ../runs/seg_smoke \
        --epochs 2 --max-volumes 2

    # 正式训练
    python train_seg.py --csv retouch_spectralis.csv --out-dir ../runs/seg_spectralis \
        --epochs 60 --batch-size 8 --input-size 224
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_lib import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_IMAGE_RANGE,
    RetouchSegDataset,
    aggregate_dice,
    build_model,
    dice_per_class,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train-seg")


def set_seed(seed: int) -> None:
    """固定随机性来源，保证划分与初始化可复现。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_volume(n_volumes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """按体积（而非切片）划分训练/验证集。

    同一体积的相邻 B-scan 高度相关，按切片随机划分会造成信息泄漏、验证分数虚高，
    因此必须按体积划分。
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_volumes)
    n_val = max(1, int(round(n_volumes * val_ratio)))
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    return train_idx, val_idx


def slices_of_volumes(dataset: RetouchSegDataset, volumes: list[int]) -> list[int]:
    """把体积下标集合展开为数据集内的切片下标集合。"""
    want = set(volumes)
    return [i for i, (vi, _) in enumerate(dataset.index) if vi in want]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[float, dict[str, float], dict[str, float]]:
    """在验证集上计算损失与逐类别 Dice。"""
    model.eval()
    per_slice: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(image)
        loss = criterion(logits, target)
        total_loss += float(loss.item())
        n_batches += 1

        for bi in range(logits.shape[0]):
            per_slice.append(dice_per_class(logits[bi], target[bi], is_logit=True))

    mean_dice = aggregate_dice(per_slice)
    arr = np.stack(per_slice, axis=0) if per_slice else np.zeros((0, len(CLASS_NAMES)))
    std = ({name: float(np.std(arr[:, ci])) for ci, name in enumerate(CLASS_NAMES)}
           if len(arr) else {name: float("nan") for name in CLASS_NAMES})
    return total_loss / max(n_batches, 1), mean_dice, std


def collate(samples) -> dict[str, torch.Tensor]:
    """把 SegSample 列表拼成 batch 张量。"""
    return {
        "image": torch.from_numpy(np.stack([s.image for s in samples])),
        "target": torch.from_numpy(np.stack([s.target for s in samples])),
    }


def train_one_fold(
    dataset: RetouchSegDataset,
    train_idx: list[int],
    val_idx: list[int],
    args,
    device: torch.device,
    fold_dir: Path,
) -> dict[str, float]:
    """训练单个 fold，返回该 fold 的最佳验证指标。"""
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )

    model = build_model(encoder_weights=args.encoder_weights, input_size=args.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 类别极不平衡（液体体素占比 < 1%），用正样本权重缓解
    pos_weight = torch.tensor([args.pos_weight] * len(CLASS_NAMES), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    fold_dir.mkdir(parents=True, exist_ok=True)
    best_dice, best_epoch = -1.0, -1
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, n_batches = 0.0, 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            run_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        val_loss, val_dice, val_std = evaluate(model, val_loader, device, criterion)
        train_loss = run_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_mean_dice": val_dice["Mean"],
               **{f"val_dice_{k}": v for k, v in val_dice.items() if k != "Mean"},
               **{f"val_std_{k}": v for k, v in val_std.items()},
               "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed}
        history.append(row)

        log.info(
            "epoch %3d/%d | train_loss %.4f | val_loss %.4f | val DSC %s | Mean %.4f | %.1fs",
            epoch, args.epochs, train_loss, val_loss,
            " ".join(f"{k}={val_dice[k]:.3f}" for k in CLASS_NAMES), val_dice["Mean"], elapsed,
        )

        if val_dice["Mean"] > best_dice:
            best_dice, best_epoch = val_dice["Mean"], epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "val_dice": val_dice, "val_std": val_std,
                 "class_names": list(CLASS_NAMES), "input_size": args.input_size,
                 "image_range": args.image_range,
                 "encoder": "resnet18", "arch": "UnetPlusPlus"},
                fold_dir / "best.pt",
            )
            log.info("  ↳ 新最优（Mean DSC %.4f），已保存 %s", best_dice, fold_dir / "best.pt")

        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    return {"best_mean_dice": best_dice, "best_epoch": best_epoch}


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 RETOUCH Spectralis 下游分割模型")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="BCE 正样本权重，应对液体体素占比 <1% 的极端不平衡")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--image-range", default=DEFAULT_IMAGE_RANGE,
                        choices=["0-1", "-1-1"],
                        help="送入分割模型的图像值域，训练与评测必须一致")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--folds", type=int, default=1,
                        help=">1 时做交叉验证（论文的 Upper 上界模型用 4-fold）")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-weights", default="imagenet",
                        help="设为 none 可离线训练（不下载预训练权重）")
    parser.add_argument("--max-volumes", type=int, default=None, help="冒烟测试用")
    parser.add_argument("--annotated-slices", type=Path, default=None,
                        help="JSON: {volume 文件名: [切片下标]}，仅保留被标注的切片")
    args = parser.parse_args()

    if args.encoder_weights.lower() in ("none", ""):
        args.encoder_weights = None

    set_seed(args.seed)
    if not torch.cuda.is_available():
        log.warning("未检测到 CUDA，将使用 CPU 训练（会非常慢）")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotated = None
    if args.annotated_slices and args.annotated_slices.exists():
        annotated = {k: set(v) for k, v in
                     json.loads(args.annotated_slices.read_text()).items()}
        log.info("已载入标注切片清单: %d 个体积", len(annotated))

    dataset = RetouchSegDataset(
        csv_path=args.csv, input_size=args.input_size,
        image_range=args.image_range,
        annotated_only=True, annotated_slices=annotated, max_volumes=args.max_volumes,
    )
    if len(dataset) == 0:
        log.error("数据集为空，请检查 CSV 与数据路径")
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    n_folds = max(1, args.folds)
    results = []
    for fold in range(n_folds):
        train_vols, val_vols = split_by_volume(len(dataset.df), args.val_ratio, args.seed + fold)
        train_idx = slices_of_volumes(dataset, train_vols)
        val_idx = slices_of_volumes(dataset, val_vols)
        log.info("=" * 70)
        log.info("Fold %d/%d | 训练 %d 体积 / %d 切片 | 验证 %d 体积 / %d 切片",
                 fold + 1, n_folds, len(train_vols), len(train_idx), len(val_vols), len(val_idx))
        log.info("  训练体积: %s", [Path(dataset.df.loc[i, 'volume']).stem for i in train_vols])
        res = train_one_fold(dataset, train_idx, val_idx, args, device,
                             args.out_dir / f"fold{fold}")
        results.append({"fold": fold, **res})

    summary = pd.DataFrame(results)
    summary.to_csv(args.out_dir / "cv_summary.csv", index=False)
    log.info("=" * 70)
    log.info("交叉验证结果:\n%s", summary.to_string(index=False))
    log.info("平均最佳 Mean DSC: %.4f (±%.4f)",
             summary["best_mean_dice"].mean(), summary["best_mean_dice"].std(ddof=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
