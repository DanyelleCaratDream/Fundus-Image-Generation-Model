# -*- coding: utf-8 -*-
"""把图像数据集居中裁成正方形（保留目录结构、保留原始分辨率）。

用来把老师给的 DR 分级数据集处理成正方形，以便后续统一 resize 到同一分辨率。
已有的 `fundus/crop_to_square.py` 是同一逻辑的写死路径版本，本脚本是通用版。

用法：
    # 处理老师数据集 -> 新目录（不动原始数据）
    python classify_project/scripts/prep_square.py \
        --src fundus/HRSG_4_758_v0.3.4 \
        --dst fundus/HRSG_4_758_square

    # 排除几何增广副本（_aug_1）
    python classify_project/scripts/prep_square.py --src ... --dst ... --drop-aug

    # 再顺手限制最长边（省磁盘、加快后续读取）
    python classify_project/scripts/prep_square.py --src ... --dst ... --max-size 1024
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def crop_one(src: Path, dst: Path, max_size: int | None,
             resize: int | None = None) -> tuple[str, int, int, int]:
    """居中裁正方形（可选限制/强制尺寸）。返回 (状态, 原宽, 原高, 输出边长)。"""
    with Image.open(src) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        size = min(w, h)

        left, top = (w - size) // 2, (h - size) // 2
        out = im.crop((left, top, left + size, top + size))

        if resize:                      # 强制统一尺寸（缩小时用 LANCZOS，放大时用 BICUBIC）
            im_res = Image.LANCZOS if size > resize else Image.BICUBIC
            out = out.resize((resize, resize), im_res)
            size = resize
        elif max_size and size > max_size:
            out = out.resize((max_size, max_size), Image.LANCZOS)
            size = max_size

        dst.parent.mkdir(parents=True, exist_ok=True)
        save_kw = {"quality": 95} if dst.suffix.lower() in (".jpg", ".jpeg") else {}
        out.save(dst, **save_kw)
    return ("ok", w, h, size)


def main() -> int:
    ap = argparse.ArgumentParser(description="居中裁正方形")
    ap.add_argument("--src", required=True, help="源目录")
    ap.add_argument("--dst", required=True, help="目标目录（会被创建）")
    ap.add_argument("--drop-aug", action="store_true",
                    help="跳过文件名含 _aug_ 的几何增广副本")
    ap.add_argument("--max-size", type=int, default=None,
                    help="裁完后若边长超过此值则缩到该值（不设则保留原始分辨率）")
    ap.add_argument("--resize", type=int, default=None,
                    help="强制把所有图统一到该边长（推荐：训练前先统一尺寸，否则每个 epoch 都要解码大图，极慢）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    if not src_root.is_dir():
        print(f"错误：源目录不存在 {src_root}", file=sys.stderr)
        return 1
    if dst_root.exists() and any(dst_root.iterdir()):
        print(f"错误：目标目录已存在且非空 {dst_root}\n"
              f"      请先删除，或换一个 --dst", file=sys.stderr)
        return 1

    files = [p for p in src_root.rglob("*")
             if p.is_file() and p.suffix.lower() in EXTS]
    if args.drop_aug:
        files = [p for p in files if "_aug_" not in p.name]
    files.sort()

    print(f"源     {src_root}")
    print(f"目标   {dst_root}")
    print(f"文件   {len(files)} 张"
          + (f"（已排除 _aug_ 副本）" if args.drop_aug else ""))
    print(f"最长边 {'限 '+str(args.max_size) if args.max_size else '保留原分辨率'}\n")

    ok = fail = 0
    sizes = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {}
        for f in files:
            rel = f.relative_to(src_root)
            futs[ex.submit(crop_one, f, dst_root / rel, args.max_size, args.resize)] = f
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                _, w, h, s = fut.result()
                ok += 1
                sizes.append(s)
                if i % 100 == 0 or i == len(files):
                    print(f"  {i}/{len(files)} ...")
            except Exception as e:
                fail += 1
                print(f"  [跳过] {f.name}: {type(e).__name__}: {e}")

    print(f"\n完成：成功 {ok}，失败 {fail}")
    if sizes:
        sizes.sort()
        print(f"边长中位 {sizes[len(sizes)//2]}  最小 {sizes[0]}  最大 {sizes[-1]}")

    # 统计目录结构，方便核对
    print("\n输出结构（各类张数）：")
    for sp in sorted(d for d in dst_root.iterdir() if d.is_dir()):
        line = f"  {sp.name:8s}"
        for cl in sorted(d for d in sp.iterdir() if d.is_dir()):
            n = sum(1 for _ in cl.glob("*") if _.is_file())
            line += f"  {cl.name}={n}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
