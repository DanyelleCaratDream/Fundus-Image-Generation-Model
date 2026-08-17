# -*- coding: utf-8 -*-
"""
poisson_gen.py —— Phase C2 阶段 2 组合 1：泊松贴片融合（跨图区域无缝搬运）

方法原理（v2，零病灶检测）：
    1) 在 512px 工作分辨率操作，最终降回 128px 输出。
    2) 随机贴片：在底图 A 视网膜内随机生成 1-3 个"病灶样"贴片 mask
       （每个贴片 = 1-3 个重叠椭圆簇，形状自然，模拟出血/渗出外观）。
    3) 跨图采样：供体 B（≠A）取一张真实图。
    4) Poisson 无缝融合：cv2.seamlessClone(NORMAL_CLONE) 把供体 B 在贴片
       区域的纹理无缝搬运到底图 A → 边缘无缝、局部光照自动适配。

为什么不需要病灶检测：
    330 张全为 KW IV 级重度眼底图，任意局部区域都含病灶/血管纹理 →
    随机贴片即"搬运病灶样内容"，无需识别病灶。病灶检测是启发式、无标定、
    无法自证（曾尝试排除血管 mask，量化验证 mask 与原始图不对齐），果断舍弃。

为什么是高价值组合：
    底图保留解剖结构 + 外来病灶样纹理 → "相似但不相同"的扩增样本，
    文献支撑（Yu et al. 2021, Biomed Opt Express；report/03-literature-fusion.md）。

致命缺点（预期，先写后验）：
    供体纹理来自真实图 → 局部仍是真实像素（记忆风险，复制检测会量化）；
    底图不变 → 输出与底图高度相似（"相似"是设计目标，但复制检测可能报警，如实上报）；
    随机贴片可能贴到供体的视盘 → 产生"多视盘"伪影（人工验收把关）。

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/poisson_gen.py --num_images 5 --montage 5      # 冒烟 + 对比蒙太奇
    python scripts/poisson_gen.py --num_images 300 --seed 42     # 全量

输出：../../eval_data/poisson/singles/sample_0000.png（128px）
     ../../eval_data/poisson/_montage/xxx.png（donor | base | output 对比条）
"""
import argparse
import os
import glob

import numpy as np
import cv2
from PIL import Image


def load_work(path, work_size):
    """加载单张素材为 work_size×work_size RGB uint8。"""
    im = Image.open(path).convert("RGB").resize((work_size, work_size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def retina_mask(img, erode_k=40):
    """底图视网膜内可放置区域：非黑像素腐蚀，避免贴片落到黑色边框。"""
    m = (img.max(axis=2) > 12).astype(np.uint8)
    return cv2.erode(m, np.ones((erode_k, erode_k), np.uint8))


def make_patch_masks(work_size, rng, retina, n_clusters, r_lo=25, r_hi=55):
    """生成 n_clusters 个"病灶样"贴片 mask（每簇 1-3 个重叠椭圆，有机形状）。

    返回 [(mask_uint8, center_xy), ...]；center 确保位于 retina 内。
    """
    patches = []
    for _ in range(n_clusters):
        center = None
        for _ in range(60):
            cx = rng.randint(0, work_size)
            cy = rng.randint(0, work_size)
            if retina[cy, cx] == 1:
                center = (cx, cy)
                break
        if center is None:
            continue
        mask = np.zeros((work_size, work_size), np.uint8)
        for _ in range(rng.randint(1, 4)):  # 1-3 个重叠椭圆
            ex = int(np.clip(center[0] + rng.randint(-10, 11), 0, work_size - 1))
            ey = int(np.clip(center[1] + rng.randint(-10, 11), 0, work_size - 1))
            rx = rng.randint(r_lo, r_hi + 1)
            ry = int(rx * rng.uniform(0.5, 1.5))
            ang = rng.randint(0, 180)
            cv2.ellipse(mask, (ex, ey), (rx, ry), ang, 0, 360, 255, -1)
        patches.append((mask, center))
    return patches


def clone_patch(base, donor, mask, center):
    """把供体 donor 在 mask 区域的纹理无缝贴到底图 base，返回 (新 base, 是否成功)。"""
    try:
        out = cv2.seamlessClone(donor, base, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return base, False
    return out, True


def main():
    ap = argparse.ArgumentParser(description="泊松贴片融合（Phase C2 阶段 2 组合 1，零检测版）")
    ap.add_argument("--data", default="../../fundus/_all_images_ORIGINAL", help="素材图目录")
    ap.add_argument("--output", default="../../eval_data/poisson", help="输出根目录")
    ap.add_argument("--work_size", type=int, default=512, help="工作分辨率（融合）")
    ap.add_argument("--img_size", type=int, default=128, help="输出分辨率")
    ap.add_argument("--num_images", type=int, default=300, help="生成数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--max_patches", type=int, default=3, help="每张最多贴片数")
    ap.add_argument("--patch_radius", type=int, default=40, help="贴片基准半径(px, work 尺度, ±50%)")
    ap.add_argument("--rot", type=float, default=0.0, help="全局旋转幅度(度, ±, 0=不变)")
    ap.add_argument("--color_jit", type=float, default=0.0, help="全局色彩抖动比例(0=不变)")
    ap.add_argument("--montage", type=int, default=0, help=">0 则生成前 N 张对比蒙太奇")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.jpg")) +
                   glob.glob(os.path.join(args.data, "*.jpeg")) +
                   glob.glob(os.path.join(args.data, "*.png")))
    if not paths:
        raise FileNotFoundError(f"目录无图像: {args.data}")
    N = len(paths)
    rng = np.random.RandomState(args.seed)
    r_lo = int(args.patch_radius * 0.6)
    r_hi = int(args.patch_radius * 1.5)
    print(f"[POISSON] seed={args.seed} 素材 {N} 张，工作分辨率 {args.work_size}，"
          f"输出 {args.img_size}，目标 {args.num_images} 张，每张贴 1-{args.max_patches} 个贴片"
          f"（半径 {r_lo}-{r_hi}px）")

    singles = os.path.join(args.output, "singles")
    os.makedirs(singles, exist_ok=True)
    montage_dir = os.path.join(args.output, "_montage")
    if args.montage > 0:
        os.makedirs(montage_dir, exist_ok=True)

    produced = 0
    attempt = 0
    while produced < args.num_images and attempt < args.num_images * 200:
        attempt += 1
        base_idx = rng.randint(N)
        base = load_work(paths[base_idx], args.work_size)
        rm = retina_mask(base)
        donor_idx = rng.randint(N)
        while donor_idx == base_idx:
            donor_idx = rng.randint(N)
        donor = load_work(paths[donor_idx], args.work_size)

        n_patches = rng.randint(1, args.max_patches + 1)
        patches = make_patch_masks(args.work_size, rng, rm, n_patches, r_lo, r_hi)
        if not patches:
            continue
        placed = 0
        for mask, center in patches:
            base, ok = clone_patch(base, donor, mask, center)
            if ok:
                placed += 1
        if placed == 0:
            continue  # 绝不保存未改动的底图（= 复制）

        # 全局微变换：降底图近复制率（v1 100%复制 → 增强后归零，见六段式汇报）
        if args.rot > 0:
            M = cv2.getRotationMatrix2D((args.work_size / 2, args.work_size / 2),
                                        rng.uniform(-args.rot, args.rot), 1.0)
            base = cv2.warpAffine(base, M, (args.work_size, args.work_size),
                                  flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        if args.color_jit > 0:
            a = base.astype(np.int16)
            for c in range(3):
                a[:, :, c] = np.clip(a[:, :, c] * rng.uniform(1 - args.color_jit, 1 + args.color_jit)
                                     + rng.uniform(-12, 12) * args.color_jit, 0, 255)
            base = a.astype(np.uint8)

        out = cv2.resize(base, (args.img_size, args.img_size), interpolation=cv2.INTER_LANCZOS4)
        Image.fromarray(out).save(os.path.join(singles, f"sample_{produced:04d}.png"))

        if args.montage > 0 and produced < args.montage:
            # 对比条：供体 | 底图 | 输出（192px 缩略，带文字标注，无框）
            THUMB = 192
            def _thumb(img, label, color=(255, 255, 255)):
                t = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_LANCZOS4)
                cv2.rectangle(t, (0, 0), (THUMB, 18), (0, 0, 0), -1)
                cv2.putText(t, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                return t
            strip = np.hstack([
                _thumb(donor, "donor(real)", (200, 200, 255)),
                _thumb(load_work(paths[base_idx], args.work_size), "base(real)", (200, 255, 200)),
                _thumb(out, "OUTPUT(generated)", (255, 200, 200)),
            ])
            Image.fromarray(strip).save(os.path.join(
                montage_dir, f"m{produced:03d}_base{base_idx:03d}_donor{donor_idx:03d}.png"))
        produced += 1
        if produced % 50 == 0:
            print(f"[POISSON] {produced}/{args.num_images}")

    print(f"[POISSON] 完成：{produced} 张（{attempt} 次尝试）→ {singles}")
    if args.montage > 0:
        print(f"[POISSON] 蒙太奇 → {montage_dir}")
    if produced < args.num_images:
        print(f"[WARN] 少于目标 {args.num_images}：请放宽 --max_patches / 检查视网膜区域")


if __name__ == "__main__":
    main()
