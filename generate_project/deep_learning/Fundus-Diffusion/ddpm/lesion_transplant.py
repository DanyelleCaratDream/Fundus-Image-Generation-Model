#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""病灶移植（D 路线）：以 D3 生成图为供体，把病灶块 Poisson 融合到 CFM@256 底图

为什么用"圆块 + 无缝融合"而不是精细分割：
- 没有病灶像素级标注，硬分割会引入假边缘；
- 圆块携带病灶及其周围纹理，Poisson 融合只改低频颜色，边界自然；
- 代价是可能带进供体的局部背景，需目检筛选。

供体：sdedit_D3_ema/orig（128，上采样到 256）
受体：judgment_20261020/CFM256_delivery_s08/singles（若不存在则用 D5_cfm256_ema/singles）
病灶打分：渗出（亮黄）+ 出血（暗红）启发式，取局部极大值、最小间距约束。

用法
    F:/python/python.exe lesion_transplant.py --num 8 --per 3
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
J = ROOT / "judgment_20261020"
DONOR_DIR = ROOT / "sdedit_D3_ema" / "t00200"
RECEP_A = J / "CFM256_delivery_s08" / "singles"
RECEP_B = J / "D5_cfm256_ema" / "singles"
OUT_DIR = J / "CFM256_lesion_transplant" / "singles"


def lesion_score(img_bgr):
    """渗出（亮黄）+ 出血（暗红）得分图；输入 uint8 BGR，输出 float32 [0,1]"""
    b, g, r = [c.astype(np.float32) / 255.0 for c in cv2.split(img_bgr)]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # 渗出：亮、黄（R/G 高、B 低）
    exud = ((r > 0.55) & (g > 0.35) & (b < 0.40) & ((r - b) > 0.18)).astype(np.float32)
    exud *= np.clip((lum - 0.45) / 0.3, 0, 1)
    # 出血：暗红（红相对绿高、整体暗）
    hemo = ((r - g) > 0.10) & (g < 0.30) & (lum < 0.35)
    hemo = hemo.astype(np.float32)
    score = cv2.GaussianBlur(np.clip(exud + 0.8 * hemo, 0, 1), (0, 0), 3.0)
    return score


def pick_centers(score, k, min_dist, margin):
    """贪心取局部极大值，避开边缘"""
    h, w = score.shape
    s = score.copy()
    s[:margin, :] = 0; s[-margin:, :] = 0; s[:, :margin] = 0; s[:, -margin:] = 0
    pts = []
    for _ in range(k):
        idx = int(np.argmax(s))
        y, x = divmod(idx, w)
        if s[y, x] <= 1e-4:
            break
        pts.append((x, y, float(s[y, x])))
        yy, xx = np.ogrid[:h, :w]
        s[(yy - y) ** 2 + (xx - x) ** 2 <= min_dist ** 2] = 0
    return pts


def transplant(dst_bgr, donor_bgr, src_xy, dst_xy, radius):
    """羽化 alpha 融合 + LAB 均值匹配（不用 seamlessClone：会把供体彩色噪声放大成彩虹点）"""
    x, y = src_xy
    qx, qy = dst_xy
    h, w = donor_bgr.shape[:2]
    x0, y0 = max(0, x - radius), max(0, y - radius)
    x1, y1 = min(w, x + radius), min(h, y + radius)
    patch = donor_bgr[y0:y1, x0:x1].astype(np.float32)
    ph, pw = patch.shape[:2]

    # 目标区域
    dx0, dy0 = max(0, qx - pw // 2), max(0, qy - ph // 2)
    dx1, dy1 = min(dst_bgr.shape[1], dx0 + pw), min(dst_bgr.shape[0], dy0 + ph)
    dst_patch = dst_bgr[dy0:dy1, dx0:dx1].astype(np.float32)
    patch = patch[:dy1 - dy0, :dx1 - dx0]

    # 羽化 mask：圆半径 0.75r + sigma 0.18r，保证方块角点 alpha≈0（否则出现方形接缝）
    mask = np.zeros((ph, pw), np.float32)
    cv2.circle(mask, (pw // 2, ph // 2), int(radius * 0.75), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), radius * 0.18)
    mask = mask[:dst_patch.shape[0], :dst_patch.shape[1]]

    # LAB 均值匹配（只改低频颜色）
    p_lab = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    d_lab = cv2.cvtColor(dst_patch.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    mm = mask > 0.5
    if mm.sum() > 50:
        for c in range(3):
            p_lab[..., c] += d_lab[..., c][mm].mean() - p_lab[..., c][mm].mean()
    patch = cv2.cvtColor(np.clip(p_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    out = dst_bgr.copy()
    a = mask[..., None]
    region = out[dy0:dy1, dx0:dx1].astype(np.float32)
    out[dy0:dy1, dx0:dx1] = np.clip(a * patch + (1 - a) * region, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=8, help="受体张数")
    ap.add_argument("--per", type=int, default=3, help="每张移植病灶块数")
    ap.add_argument("--radius", type=int, default=24, help="病灶块半径（256 口径）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--donor_dir", default=str(DONOR_DIR),
                    help="供体目录；可换老师真实 severe 对照")
    ap.add_argument("--out_tag", default="d3donor", help="输出子目录标签")
    a = ap.parse_args()

    recep_dir = RECEP_A if RECEP_A.exists() and any(RECEP_A.glob("*.png")) else RECEP_B
    donor_dir = Path(a.donor_dir)
    donors = sorted(list(donor_dir.glob("*.png")) + list(donor_dir.glob("*.jpg")))
    receps = sorted(recep_dir.glob("*.png"))[:a.num]
    assert donors and receps, f"donor={donor_dir} recep={recep_dir}"
    OUT_DIR = J / "CFM256_lesion_transplant" / a.out_tag
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # 供体预处理：128 → 256，记录候选病灶中心
    donor_data = []
    for f in donors:
        bgr = cv2.cvtColor(np.array(Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
        sc = lesion_score(bgr)
        pts = pick_centers(sc, 5, min_dist=40, margin=40)
        if pts:
            donor_data.append((bgr, pts))
    print(f"[donor] {len(donor_data)}/{len(donors)} 有候选病灶", flush=True)

    for i, rf in enumerate(receps):
        dst = cv2.cvtColor(np.array(Image.open(rf).convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = dst.shape[:2]
        used = 0
        for _ in range(a.per):
            dbgr, pts = donor_data[int(rng.integers(len(donor_data)))]
            sx, sy, _s = pts[int(rng.integers(len(pts)))]
            # 受体位置：眼底圆内（半径 0.42·min），避开视盘中心附近
            for _try in range(20):
                rr = 0.42 * min(w, h) * np.sqrt(rng.uniform(0, 1))
                th = rng.uniform(0, 2 * np.pi)
                qx = int(w / 2 + rr * np.cos(th))
                qy = int(h / 2 + rr * np.sin(th))
                if (qx - w / 2) ** 2 + (qy - h / 2) ** 2 < (0.42 * min(w, h)) ** 2:
                    break
            dst = transplant(dst, dbgr, (sx, sy), (qx, qy), a.radius)
            used += 1
        Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)).save(OUT_DIR / rf.name)
        print(f"[ok] {rf.name} transplants={used}", flush=True)

    # 对照表：受体原图 / 移植后
    rows = [(sorted(recep_dir.glob("*.png"))[:a.num], "CFM@256 + SDEdit 0.8 (before)"),
            (sorted(OUT_DIR.glob("*.png"))[:a.num], "after lesion transplant")]
    S, PAD, CAP, N = 256, 6, 24, a.num
    canvas = Image.new("RGB", (N * S + (N + 1) * PAD, len(rows) * (S + CAP + PAD) + PAD), "white")
    dr = ImageDraw.Draw(canvas)
    y = PAD
    for files, label in rows:
        dr.text((PAD, y), label, fill="black"); y += CAP
        for j, f in enumerate(files[:N]):
            canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC), (PAD + j * (S + PAD), y))
        y += S + PAD
    out = J / f"lesion_transplant_compare_{a.out_tag}.png"
    canvas.save(out)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
