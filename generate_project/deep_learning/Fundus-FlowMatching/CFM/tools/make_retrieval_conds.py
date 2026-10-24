#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FSFM 式检索 + 重心采样：从 severe anchor 的 k 近邻单纯形里采样条件向量

给定手工/RETFound 特征：
  1. 标准化 + L2 归一化；
  2. 对每个条件向量：随机选一个 severe anchor，取其 k 个类内近邻；
  3. 权重 ~ Dirichlet(1)，条件 = Σ w_i z_i；
  4. 保存 [M, D] 到 <feat_dir>/conds_severe.npy。

用法
    python tools/make_retrieval_conds.py --feat_dir data/handcrafted --num 64
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEVERE_DIR = Path(r"D:/AI_Model_Project_for_Fundus_Color_Images/fundus/HRSG_4_758_sq256/train/severe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = Path(a.feat_dir)
    X = np.load(d / "feats.npy")
    names = json.load(open(d / "names.json", encoding="utf-8"))
    sev = {f.stem for f in SEVERE_DIR.iterdir() if "_aug_1" not in f.name}
    labels = np.array([1 if Path(n).stem in sev else 0 for n in names])

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    sim = Xs @ Xs.T

    sev_idx = np.where(labels == 1)[0]
    rng = np.random.default_rng(a.seed)
    conds = []
    for _ in range(a.num):
        anchor = int(rng.choice(sev_idx))
        cand = [j for j in sev_idx if j != anchor]
        order = sorted(cand, key=lambda j: -sim[anchor, j])[:a.k]
        idxs = [anchor] + order
        w = rng.dirichlet(np.ones(len(idxs)))
        conds.append((Xs[idxs] * w[:, None]).sum(0))
    conds = np.stack(conds).astype(np.float32)
    np.save(d / "conds_severe.npy", conds)
    print(f"[done] {conds.shape} -> {d/'conds_severe.npy'}")


if __name__ == "__main__":
    main()
