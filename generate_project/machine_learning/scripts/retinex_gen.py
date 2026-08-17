# -*- coding: utf-8 -*-
"""
retinex_gen.py —— Phase C2 阶段 2 组合 2：Retinex 光照交换

方法原理：
    1) Retinex 分解（经典单尺度，log 空间）：
       log(图) = log(光照) + log(反射)
       - 光照 L = log 域高斯低通（大核，平滑明暗渐变）
       - 反射 R = log 域细节（结构/病灶/血管/颜色）
       验证：L*R 完美重构原图（误差 0）。
    2) 跨图交换光照：底图 A 保留反射 R_A，供体 B 提供光照 L_B，
       重组成 新图 = R_A · L_B^α（α∈[0,1] 控制交换强度，半混合）。
    3) 降回 128px 输出。

为什么是高价值组合：
    光照承载采集时的明暗/色温，反射承载诊断语义（血管/视盘/病灶）。
    交换光照 = 结构语义不变、全局光照风格变 → "相似但不相同"的安全多样性放大器。
    文献：Zhang, Li & Shin (2022, Comput Biol Med 145:105427) 无监督 Retinex
    分解(ID-Net)只增强光照层、保反射层 → 眼底分割超 SOTA SDG 9.6% Dice；
    本脚本为其无需网络的经典改编（跨图交换而非网络随机化）。

关键设计决策（原型验证，2026-08-03）：
    半混合 α≈0.7 使复制率归零（纯交换 7% → α0.7 0%）。大核 k41 保持光照平滑、
    纹理全在反射里（小核 k25 把纹理混进光照 → 交换携带结构 → 复制率飙到 27%）。

致命缺点（预期，先写后验）：
    光照交换是全局变化，若供体光照过暗/过亮 → 输出可能偏暗/偏亮（亮度越界）；
    每张的α随机 → 多样性有保证但强度不可控；病灶/血管语义不变 → 与底图解剖相同。

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/retinex_gen.py --num_images 5 --montage 5     # 冒烟 + 对比蒙太奇
    python scripts/retinex_gen.py --num_images 300 --seed 42    # 全量

输出：../../eval_data/retinex/singles/sample_0000.png（128px）
     ../../eval_data/retinex/_montage/xxx.png（donor | base | output 对比条）
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


def retinex_decomp(img, kernel):
    """经典单尺度 Retinex 分解（log 空间）。

    返回 (logL, logR)：光照与反射的 log 域表示。
    img: uint8 [H,W,3]（RGB）。
    """
    I = img.astype(np.float64) + 1.0  # 防 log(0)
    logI = np.log(I)
    logL = cv2.GaussianBlur(logI, (kernel, kernel), 0)  # 光照 = 平滑渐变
    logR = logI - logL                                   # 反射 = 细节
    return logL, logR


def exchange(base, donor, kernel=41, alpha=0.8):
    """跨图光照交换：底图 A 的反射 × 供体 B 的光照（α 混合强度）。

    新图 = exp(α·logL_B + logR_A) - 1 = R_A · L_B^α
    α=1 纯交换；α→0 趋近底图。返回 uint8 [H,W,3]。
    """
    logL_B, _ = retinex_decomp(donor, kernel)
    _, logR_A = retinex_decomp(base, kernel)
    new = np.exp(alpha * logL_B + logR_A) - 1.0
    return np.clip(new, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Retinex 光照交换（Phase C2 阶段 2 组合 2）")
    ap.add_argument("--data", default="../../fundus/_all_images_ORIGINAL", help="素材图目录")
    ap.add_argument("--output", default="../../eval_data/retinex", help="输出根目录")
    ap.add_argument("--work_size", type=int, default=512, help="工作分辨率（分解+交换）")
    ap.add_argument("--img_size", type=int, default=128, help="输出分辨率")
    ap.add_argument("--num_images", type=int, default=300, help="生成数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--kernel", type=int, default=41, help="光照低通核(px, 越大光照越纯)")
    ap.add_argument("--alpha_lo", type=float, default=0.6, help="交换强度下限")
    ap.add_argument("--alpha_hi", type=float, default=0.9, help="交换强度上限")
    ap.add_argument("--diag", type=int, default=6, help="诊断：打印前 N 张分解统计")
    ap.add_argument("--montage", type=int, default=0, help=">0 则生成前 N 张对比蒙太奇")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.jpg")) +
                   glob.glob(os.path.join(args.data, "*.jpeg")) +
                   glob.glob(os.path.join(args.data, "*.png")))
    if not paths:
        raise FileNotFoundError(f"目录无图像: {args.data}")
    N = len(paths)
    rng = np.random.RandomState(args.seed)
    print(f"[RETINEX] seed={args.seed} 素材 {N} 张，工作分辨率 {args.work_size}，"
          f"输出 {args.img_size}，目标 {args.num_images} 张，α∈[{args.alpha_lo},{args.alpha_hi}]，核 {args.kernel}")

    # ---- 诊断：验证分解 ----
    if args.diag > 0:
        for idx in range(min(args.diag, N)):
            img = load_work(paths[idx], args.work_size)
            logL, _ = retinex_decomp(img, args.kernel)
            L = np.exp(logL)
            b = img.max(axis=2)
            print(f"[DIAG] img {idx}: 光照L均值(视网膜内) {L[b>12].mean():.0f}, "
                  f"亮度 {b[b>12].mean():.0f}")

    singles = os.path.join(args.output, "singles")
    os.makedirs(singles, exist_ok=True)
    montage_dir = os.path.join(args.output, "_montage")
    if args.montage > 0:
        os.makedirs(montage_dir, exist_ok=True)

    for i in range(args.num_images):
        base_idx = rng.randint(N)
        donor_idx = rng.randint(N)
        while donor_idx == base_idx:
            donor_idx = rng.randint(N)
        base = load_work(paths[base_idx], args.work_size)
        donor = load_work(paths[donor_idx], args.work_size)
        alpha = rng.uniform(args.alpha_lo, args.alpha_hi)
        out = exchange(base, donor, args.kernel, alpha)
        out = cv2.resize(out, (args.img_size, args.img_size), interpolation=cv2.INTER_LANCZOS4)
        Image.fromarray(out).save(os.path.join(singles, f"sample_{i:04d}.png"))

        if args.montage > 0 and i < args.montage:
            THUMB = 192
            def _thumb(img, label, color=(255, 255, 255)):
                t = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_LANCZOS4)
                cv2.rectangle(t, (0, 0), (THUMB, 18), (0, 0, 0), -1)
                cv2.putText(t, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                return t
            strip = np.hstack([
                _thumb(donor, "donor(real)", (200, 200, 255)),
                _thumb(base, "base(real)", (200, 255, 200)),
                _thumb(out, "OUTPUT(generated)", (255, 200, 200)),
            ])
            Image.fromarray(strip).save(os.path.join(
                montage_dir, f"m{i:03d}_base{base_idx:03d}_donor{donor_idx:03d}_a{alpha:.2f}.png"))
        if (i + 1) % 50 == 0:
            print(f"[RETINEX] {i + 1}/{args.num_images}")

    print(f"[RETINEX] 完成：{args.num_images} 张 → {singles}")
    if args.montage > 0:
        print(f"[RETINEX] 蒙太奇 → {montage_dir}")


if __name__ == "__main__":
    main()
