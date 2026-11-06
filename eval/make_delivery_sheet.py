#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交付图筛选包：200 张 → 编号缩略图总表 + 索引 CSV

- 复用 metrics_fundus.lesion_fractions 计算疑似渗出/出血像素占比（颜色阈值代理，未验证）；
- 按渗出占比降序编号 1..N，每页 25 张（5×5），另出一张 top80；
- `--pick 12,45,...` 把选中序号复制到 picked/。

用法
    python eval/make_delivery_sheet.py
    python eval/make_delivery_sheet.py --pick 1,7,23
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from metrics_fundus import lesion_fractions  # noqa: E402

FONT = "C:/Windows/Fonts/msyh.ttc"


def load_rgb01(paths, size=256):
    arr = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        arr.append(np.asarray(im, np.float32) / 255.0)
    return np.stack(arr)


def sheet(items, cols, rows, cell, out, title, font, font_s):
    PAD, CAP = 6, 22
    W = cols * (cell + PAD) + PAD
    H = CAP + rows * (cell + PAD + CAP) + PAD
    cv = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cv)
    d.text((PAD, 3), title, fill="black", font=font)
    for k, (idx, path, exud) in enumerate(items):
        r, c = divmod(k, cols)
        x = PAD + c * (cell + PAD)
        y = CAP + r * (cell + PAD + CAP)
        cv.paste(Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y))
        d.text((x, y + cell + 2), f"#{idx:03d}  渗出 {exud:.3f}", fill="black", font=font_s)
    cv.save(out)
    print("[sheet]", out, cv.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "eval_data/delivery_final/singles"))
    ap.add_argument("--out", default=str(REPO / "judgment/delivery_screening_20261106"))
    ap.add_argument("--pick", default="", help="逗号分隔的序号，复制到 picked/")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    assert files, f"no png in {src}"

    arr = load_rgb01(files, 256)
    hemo, exud = lesion_fractions(arr)
    order = np.argsort(-exud)  # 渗出降序
    items = []
    rows = []
    for rank, i in enumerate(order, start=1):
        items.append((rank, files[i], float(exud[i])))
        rows.append({"index": rank, "file": files[i].name,
                     "exudate_proxy": round(float(exud[i]), 5),
                     "hemorrhage_proxy": round(float(hemo[i]), 5),
                     "rgb_mean": [round(float(v), 4) for v in arr[i].mean((0, 1))]})
    with open(out / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    font = ImageFont.truetype(FONT, 20)
    font_s = ImageFont.truetype(FONT, 13)
    per = 25
    for page in range((len(items) + per - 1) // per):
        chunk = items[page * per:(page + 1) * per]
        sheet(chunk, 5, 5, 200, out / f"sheet_{page + 1:02d}.png",
              f"交付图筛选 第 {page + 1} 页（按疑似渗出占比降序；代理指标未验证）", font, font_s)
    sheet(items[:80], 10, 8, 150, out / "top80.png",
          "疑似渗出占比最高的 80 张（代理指标，未验证；仅用于初筛）", font, font_s)

    if a.pick:
        pick_dir = out / "picked"
        pick_dir.mkdir(exist_ok=True)
        want = {int(x) for x in a.pick.split(",") if x.strip()}
        n = 0
        for idx, path, _ in items:
            if idx in want:
                shutil.copy2(path, pick_dir / f"{idx:03d}_{path.name}")
                n += 1
        print(f"[pick] {n} -> {pick_dir}")

    print(f"[done] {len(items)} 张；索引 {out / 'index.csv'}")


if __name__ == "__main__":
    main()
