# -*- coding: utf-8 -*-
"""DR 四级分级分类器 —— 用于分辨率消融实验（128 / 256 / 512 哪个够用）。

设计取舍（为了让"分辨率"成为唯一变量）：
  · 训练时**不做数据增强**，只 resize + 归一化
  · 固定用数据集自带的 train/val/test 划分（不重新切分，保持与老师协议一致）
  · 多个随机种子重复训练，报告 均值 ± 标准差

用法：
    python classify_project/scripts/train_classifier.py --data fundus/HRSG_4_758_square --size 256
    python classify_project/scripts/train_classifier.py --data ... --size 256 --seeds 42 52 62
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

Image.MAX_IMAGE_PIXELS = None
CLASSES = ["normal", "mild", "moderate", "severe"]   # 有序！0<1<2<3
NUM_CLASSES = len(CLASSES)


class FundusFolder(Dataset):
    """按 {root}/{split}/{class}/*.jpg 读取。"""

    def __init__(self, root: Path, split: str, size: int, train: bool):
        self.items: list[tuple[Path, int]] = []
        for ci, cname in enumerate(CLASSES):
            d = root / split / cname
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.items.append((f, ci))
        self.size = size
        # 训练/评估用同一套预处理（只是缩放+归一化），保证分辨率是唯一变量
        self.tf = transforms.Compose([
            transforms.Resize((size, size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        return self.tf(im), label


def build_model(size: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    severe = CLASSES.index("severe")
    rec = {}
    for i, c in enumerate(CLASSES):
        m = y_true == i
        rec[f"recall_{c}"] = float((y_pred[m] == i).mean()) if m.any() else float("nan")
    metrics = {
        "acc": float((y_true == y_pred).mean()),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "severe_recall": rec["recall_severe"],
        "n": int(len(y_true)),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)).tolist(),
    }
    return metrics, y_true, y_pred


def train_one(size: int, root: Path, seed: int, epochs: int, bs: int,
              lr: float, weight_decay: float, device: str, weighted: bool,
              num_workers: int, quiet: bool) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tr = FundusFolder(root, "train", size, train=True)
    va = FundusFolder(root, "val", size, train=False)
    te = FundusFolder(root, "test", size, train=False)
    if not tr.items or not te.items:
        raise RuntimeError(f"数据为空：train={len(tr)} test={len(te)}")

    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=num_workers,
                       pin_memory=True, drop_last=False)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(te, batch_size=bs, shuffle=False, num_workers=num_workers)

    model = build_model(size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if weighted:
        cnt = np.bincount([y for _, y in tr.items], minlength=NUM_CLASSES).astype(np.float32)
        w = torch.tensor((cnt.sum() / (cnt + 1e-6)) / NUM_CLASSES, device=device)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()

    best = {"qwk": -2.0}
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y)
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            m, _, _ = evaluate(model, dl_va, device) if len(va) else ({"qwk": 0}, None, None)
            if not quiet:
                print(f"      seed{seed} ep{ep:3d}  train_loss {tot/len(tr):.4f}  val_qwk {m['qwk']:.4f}")
            if m["qwk"] > best["qwk"]:
                best = m
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    te_m, y_true, y_pred = evaluate(model, dl_te, device)
    te_m["seed"] = seed
    te_m["val_qwk"] = best["qwk"]
    te_m["train_n"] = len(tr)
    te_m["test_n"] = len(te)
    return te_m


def main() -> int:
    ap = argparse.ArgumentParser(description="DR 分级分类器（分辨率消融）")
    ap.add_argument("--data", required=True, help="正方形数据集根目录")
    ap.add_argument("--size", type=int, required=True, help="输入分辨率")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--no-weighted", action="store_true", help="不加类别权重")
    ap.add_argument("--no-pretrained", action="store_true", help="不用 ImageNet 预训练")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        print(f"错误：数据目录不存在 {root}\n先跑 prep_square.py", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weighted = not args.no_weighted

    print(f"{'='*66}")
    print(f"  分辨率 {args.size}×{args.size}   设备 {device}   种子 {args.seeds}")
    print(f"  数据 {root}")
    print(f"{'='*66}")

    runs = []
    for s in args.seeds:
        t0 = time.time()
        try:
            r = train_one(args.size, root, s, args.epochs, args.batch_size, args.lr,
                          args.weight_decay, device, weighted, args.num_workers, args.quiet)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  seed{s} ❌ 显存不足，跳过")
                torch.cuda.empty_cache()
                continue
            raise
        runs.append(r)
        print(f"  seed{s:<3d} → acc {r['acc']:.4f}  bAcc {r['balanced_acc']:.4f}  "
              f"macroF1 {r['macro_f1']:.4f}  QWK {r['qwk']:.4f}  "
              f"severe召回 {r['severe_recall']:.4f}   ({time.time()-t0:.0f}s)")

    if not runs:
        print("  全部失败")
        return 2

    def agg(k):
        v = np.array([r[k] for r in runs], dtype=float)
        return float(v.mean()), float(v.std())

    summary = {k: agg(k) for k in ("acc", "balanced_acc", "macro_f1", "qwk", "severe_recall")}
    print(f"\n  ── 分辨率 {args.size} 汇总（{len(runs)} 个种子）──")
    for k, (m, s) in summary.items():
        print(f"     {k:16s} {m:.4f} ± {s:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"size": args.size, "seeds": args.seeds, "weighted": weighted,
                       "pretrained": not args.no_pretrained, "epochs": args.epochs,
                       "summary": {k: {"mean": m, "std": s} for k, (m, s) in summary.items()},
                       "runs": runs}, f, ensure_ascii=False, indent=2)
        print(f"\n  已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
