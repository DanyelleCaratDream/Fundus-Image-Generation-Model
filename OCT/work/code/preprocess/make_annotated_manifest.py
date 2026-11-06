#!/usr/bin/env python3
"""生成 annotated_slices.json：每个体积中被人工标注的 B-scan 下标。

RETOUCH 训练集的 reference 标注覆盖全部 B-scan（无病灶的切片是空掩膜，但仍是
“已标注”）。若没有显式清单，下游代码会退化为“跳过全零掩膜”，从而把**真阴性切片**
也丢掉，DSC 口径与论文不一致。因此这里显式列出每个体积的全部切片下标。

如果将来拿到 RETOUCH 官方的逐切片标注清单，可用它替换本文件的输出（键为体积文件名）。

用法
    python preprocess/make_annotated_manifest.py \
        --npy-root data/retouch --out data/retouch/annotated_slices.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEVICES = ["spectralis", "cirrus", "topcon"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.npy_root)
    manifest: dict[str, list[int]] = {}
    for dev in DEVICES:
        ann_dir = root / "annotations" / dev
        if not ann_dir.is_dir():
            print(f"[skip] 缺少 {ann_dir}")
            continue
        files = sorted(ann_dir.glob("*.npy"))
        for f in files:
            arr = np.load(f, mmap_mode="r")
            manifest[f"{dev}/{f.name}"] = list(range(int(arr.shape[0])))
        print(f"[ok] {dev}: {len(files)} volumes")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(manifest)} volumes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
