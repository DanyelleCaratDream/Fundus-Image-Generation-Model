#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用无偏 KID 实现重算已有模型的 KID（保留旧值，不静默覆盖）

背景（2026-11-06）：旧 compute_kid 的子集自助是死代码，实际用含对角的全矩阵
有偏估计且无方差。metrics_common.py 已改无偏子集自助。本脚本对已有采样目录重算，
同时保留旧值（kid_old）与来源校验（kid_old_recomputed）。

用法
    python eval/recompute_kid.py --device cpu            # dry-run，只看表
    python eval/recompute_kid.py --device cpu --apply    # 写回 *_metrics.json
输出
    eval_data/_audit/kid_recompute_2026-11-06.json
    eval_data/_audit/kid_recompute_2026-11-06.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from metrics_common import (  # noqa: E402
    load_images, InceptionFeatures, compute_kid_stats, polynomial_kernel,
)


def old_kid_bias(real_feats, fake_feats, degree=3, coef0=1.0):
    """旧实现：全矩阵、含对角、无方差（仅用于来源校验，不用于新结果）。"""
    kk = polynomial_kernel(real_feats, real_feats, degree, coef0)
    ll = polynomial_kernel(fake_feats, fake_feats, degree, coef0)
    kl = polynomial_kernel(real_feats, fake_feats, degree, coef0)
    return float(np.mean(kk) + np.mean(ll) - 2 * np.mean(kl))


def feats_for(dir_path, inc, cache, img_size=256):
    if cache.exists():
        return np.load(cache)["f"]
    imgs = load_images(str(dir_path), img_size)
    if imgs is None:
        raise FileNotFoundError(dir_path)
    f = inc(imgs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, f=f)
    return f


MODELS = ["cfm128", "cfm256", "cfm256_lesion", "cfm128_align", "cfm128_text",
          "cfm128_notime", "cfm128_featC", "cfm128_featA", "d3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--real", default=str(REPO / "eval_data/real_severe_train"))
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--apply", action="store_true", help="写回 *_metrics.json")
    a = ap.parse_args()

    audit = REPO / "eval_data/_audit"
    cache_dir = audit / "inception_feats"
    inc = InceptionFeatures(a.device)
    real_f = feats_for(Path(a.real), inc, cache_dir / "real_severe_train.npz", a.img_size)

    rows = []
    for tag in a.models:
        fake_dir = REPO / "eval_data" / tag / "singles"
        mpath = REPO / "eval_data" / f"{tag}_metrics.json"
        if not fake_dir.exists() or len(list(fake_dir.glob("*.png"))) == 0:
            print(f"[skip] {tag}: no singles")
            continue
        fake_f = feats_for(fake_dir, inc, cache_dir / f"{tag}.npz", a.img_size)
        kid_new, kid_std = compute_kid_stats(real_f, fake_f, seed=0)
        old_re = old_kid_bias(real_f, fake_f)
        old_stored = None
        if mpath.exists():
            d = json.loads(mpath.read_text(encoding="utf-8"))
            old_stored = d.get("kid")
        rows.append({"model": tag, "kid_old": old_stored,
                     "kid_old_recomputed": round(old_re, 6),
                     "kid": round(kid_new, 6), "kid_std": round(kid_std, 6),
                     "n_real": len(real_f), "n_fake": len(fake_f)})
        print(f"[kid] {tag}: old={old_stored} (re={old_re:.4f}) -> new={kid_new:.6f} ± {kid_std:.6f}")

        if a.apply and mpath.exists():
            d = json.loads(mpath.read_text(encoding="utf-8"))
            if "kid_old" not in d:
                d["kid_old"] = old_stored
            d["kid"] = float(kid_new)
            d["kid_std"] = float(kid_std)
            d["kid_impl"] = "unbiased_subset_2026-11-06"
            mpath.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    audit.mkdir(parents=True, exist_ok=True)
    (audit / "kid_recompute_2026-11-06.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(audit / "kid_recompute_2026-11-06.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)

    print("\n| 模型 | KID_old（有偏） | KID_new（无偏） | std |")
    print("|:--|--:|--:|--:|")
    for r in rows:
        print(f"| {r['model']} | {r['kid_old']} | {r['kid']} | {r['kid_std']} |")
    print(f"\n[done] -> {audit / 'kid_recompute_2026-11-06.json'}"
          + ("" if a.apply else "  (dry-run; use --apply to patch JSON)"))


if __name__ == "__main__":
    main()
