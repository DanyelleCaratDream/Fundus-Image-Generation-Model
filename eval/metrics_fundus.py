"""
专用层评估指标（自设计，回应老师 Bug 1 的"自设计指标"要求）
===========================================================
针对眼底彩照生成专门设计的指标，是通用层（metrics_common.py）之下的第二层。
这批指标在 Phase A 初版被以"难评"为由砍掉，现在补齐跑掉。

指标清单：
  病灶类（颜色阈值，暗红=出血 / 亮黄白=渗出，无标注下的近似代理）:
    hemo/exud 每图像面积占比分布 -> 真实 vs 生成 Wasserstein 距离 + 保留率
  血管类（skimage Frangi 血管度，比 tophat 管线鲁棒）:
    血管面积占比分布 -> Wasserstein 距离
    Vessel Dice（仅条件模型，对生成时使用的输入血管 mask）
  相似性 / 记忆检测:
    每张生成图对真实集的最近邻 SSIM（验证"相似但不相同"定位）
  C2ST 真伪分类 AUC（小 CNN，5 折交叉验证；值越接近 1.0 越容易被识破）
  无参考质量 BRISQUE（piq；注意在自然图像上训练，眼底图仅供参考）

用法:
    python metrics_fundus.py --real <真实图目录> --fake <生成图目录> \
        [--model <模型名>] [--cond_path <conditions 目录>] [--img_size 128] \
        [--device cuda] [--skip_c2st] [--c2st_epochs 25] [--json]

输出: JSON（含全部专用层指标）。
依赖: torch, pytorch_fid, pytorch_msssim, scipy, sklearn, skimage, cv2, piq
"""

import argparse
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.filters import frangi
from scipy.stats import wasserstein_distance

# 复用通用层的加载器与 Inception 特征提取
from metrics_common import load_images, InceptionFeatures

# ---- 病灶颜色阈值（在真实图上标定，见 Phase A 阈值调试记录）----
HEMO_R_G_MIN = 0.04   # R - G 至少
HEMO_R_B_MIN = 0.04   # R - B 至少
HEMO_R_MAX = 0.42     # R 小于此（暗红）
EXUD_MIN_RGB = 0.42   # 三通道最小值至少（亮）
EXUD_MEAN_RGB = 0.48  # 三通道均值至少（接近白/黄）
VESSEL_FRANGI_THR = 0.05   # frangi 血管度阈值（算面积占比）
VESSEL_SIGMAS = (1, 3)     # frangi 尺度

# ---- 记忆检测阈值 ----
MEM_SSIM_DUP_THR = 0.85    # 最近邻 SSIM 超过此值视为"接近复制"


# ------------------------------------------------------------
# 图像加载（返回 [N,3,128,128] 于 [-1,1]）
# ------------------------------------------------------------
def load_tensor(folder, img_size=128):
    return load_images(folder, img_size)


def to_numpy_rgb01(images):
    """[-1,1] torch -> [N,H,W,3] float 0-1 numpy（病灶/血管用）。"""
    return ((images.permute(0, 2, 3, 1) + 1.0) / 2.0).clamp(0, 1).cpu().numpy()


# ------------------------------------------------------------
# 病灶：出血 / 渗出 颜色阈值
# ------------------------------------------------------------
def lesion_masks(rgb01):
    """rgb01: [N,H,W,3] float 0-1。返回 (hemo_mask, exud_mask) 同形状 bool。"""
    R = rgb01[..., 0]
    G = rgb01[..., 1]
    B = rgb01[..., 2]
    hemo = (R - G > HEMO_R_G_MIN) & (R - B > HEMO_R_B_MIN) & (R < HEMO_R_MAX)
    mn = np.minimum(np.minimum(R, G), B)
    exud = (mn > EXUD_MIN_RGB) & ((R + G + B) / 3 > EXUD_MEAN_RGB)
    return hemo, exud


def lesion_fractions(rgb01):
    hemo, exud = lesion_masks(rgb01)
    n = rgb01.shape[0]
    hemo_frac = hemo.reshape(n, -1).mean(axis=1)
    exud_frac = exud.reshape(n, -1).mean(axis=1)
    return hemo_frac, exud_frac


# ------------------------------------------------------------
# 血管：Frangi 血管度
# ------------------------------------------------------------
def vessel_fractions(rgb01, batch=64):
    """每张图的血管面积占比（frangi > VESSEL_FRANGI_THR）。"""
    n = rgb01.shape[0]
    fracs = np.zeros(n)
    for i in range(0, n, batch):
        for j in range(i, min(i + batch, n)):
            g = rgb01[j, :, :, 1]  # 绿色通道
            v = frangi(g, sigmas=VESSEL_SIGMAS, black_ridges=True)
            fracs[j] = (v > VESSEL_FRANGI_THR).mean()
    return fracs


def dice_coef(a, b):
    """a, b: bool 数组。Dice = 2|A∩B| / (|A|+|B|)。"""
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    if s == 0:
        return float("nan")
    return 2.0 * inter / s


# ------------------------------------------------------------
# 记忆检测：生成图对真实集最近邻 SSIM
# ------------------------------------------------------------
def compute_memorization(real_images, fake_images, device="cuda"):
    import pytorch_msssim
    inc = InceptionFeatures(device)
    r_f = inc(real_images)
    f_f = inc(fake_images)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(r_f)
    _, nn_idx = nn.kneighbors(f_f)
    nn_idx = nn_idx[:, 0]

    ssim_fn = pytorch_msssim.SSIM(data_range=2.0, size_average=False)
    ssims = []
    with torch.no_grad():
        r = real_images.to(device)
        f = fake_images.to(device)
        for i in range(0, len(f), 16):
            fi = f[i:i + 16]
            ri = r[nn_idx[i:i + 16]]
            s = ssim_fn(fi, ri).cpu().numpy()  # [B]
            ssims.append(s)
    ssims = np.concatenate(ssims)
    return {
        "mem_ssim_mean": float(ssims.mean()),                     # 平均最近邻 SSIM
        "mem_ssim_median": float(np.median(ssims)),
        "mem_ssim_pct_gt_085": float((ssims > MEM_SSIM_DUP_THR).mean()),  # "接近复制"占比
    }


# ------------------------------------------------------------
# Vessel Dice：条件模型生成图 vs 输入血管 mask
# ------------------------------------------------------------
def compute_vessel_dice(fake_images, cond_path, device="cuda"):
    """sample_i ↔ sorted conditions[i % 330]（generate.py 的循环对应）。"""
    exts = ["*.png", "*.jpg", "*.jpeg"]
    import glob
    cond_files = []
    for e in exts:
        cond_files.extend(glob.glob(os.path.join(cond_path, e)))
    cond_files = sorted(cond_files)
    if not cond_files:
        return None

    rgb01 = to_numpy_rgb01(fake_images)
    dices = []
    for i in range(len(rgb01)):
        cond = np.array(Image.open(cond_files[i % len(cond_files)]).convert("L")) > 0
        g = rgb01[i, :, :, 1]
        v = frangi(g, sigmas=VESSEL_SIGMAS, black_ridges=True)
        fake_vessel = v > VESSEL_FRANGI_THR
        d = dice_coef(fake_vessel, cond)
        if not np.isnan(d):
            dices.append(d)
    return float(np.mean(dices)) if dices else None


# ------------------------------------------------------------
# C2ST：真伪二分类 AUC（小 CNN，5 折交叉验证）
# ------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        self.net = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_c2st(real_images, fake_images, device="cuda", epochs=25, seed=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = torch.cat([real_images, fake_images], dim=0)
    y = np.array([0] * len(real_images) + [1] * len(fake_images))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        model = SmallCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt, yt = X[tr_idx].to(device), torch.tensor(y[tr_idx], dtype=torch.float32).to(device)
        Xv = X[va_idx].to(device)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xt), device=device)
            for b in range(0, len(Xt), 32):
                idx = perm[b:b + 32]
                logits = model(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xv)).cpu().numpy()
        aucs.append(roc_auc_score(y[va_idx], probs))

    return {
        "c2st_auc": float(np.mean(aucs)),          # 越高=越容易被识破（越差）
        "c2st_auc_std": float(np.std(aucs)),
    }


# ------------------------------------------------------------
# BRISQUE（无参考质量，piq；自然图像训练，眼底图仅供参考）
# ------------------------------------------------------------
def compute_brisque(images, device="cuda"):
    try:
        import piq
        with torch.no_grad():
            x = ((images + 1.0) / 2.0).clamp(0, 1).to(device)
            score = piq.brisque(x, data_range=1.0).item()
        return float(score)
    except Exception as e:
        return None


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def evaluate(real_dir, fake_dir, model=None, cond_path=None, img_size=128,
             device="cuda", skip_c2st=False, c2st_epochs=25, verbose=True):
    import sys as _sys
    real = load_images(real_dir, img_size)
    fake = load_images(fake_dir, img_size)
    if real is None or fake is None:
        return None
    if verbose:
        print(f"[DATA] real={len(real)} fake={len(fake)}", file=_sys.stderr)

    results = {}

    # --- 病灶 ---
    r_rgb = to_numpy_rgb01(real)
    f_rgb = to_numpy_rgb01(fake)
    r_hemo, r_exud = lesion_fractions(r_rgb)
    f_hemo, f_exud = lesion_fractions(f_rgb)
    results["lesion_hemo_wass"] = float(wasserstein_distance(r_hemo, f_hemo))
    results["lesion_exud_wass"] = float(wasserstein_distance(r_exud, f_exud))
    r_hemo_mean = float(r_hemo.mean())
    results["lesion_hemo_retention"] = float(f_hemo.mean() / r_hemo_mean) if r_hemo_mean > 0 else None
    r_exud_mean = float(r_exud.mean())
    results["lesion_exud_retention"] = float(f_exud.mean() / r_exud_mean) if r_exud_mean > 0 else None
    # 图像级"病灶保留率"：生成图含病灶(≥0.5×真实中位)的比例
    thr_hemo = 0.5 * np.median(r_hemo)
    results["lesion_hemo_presence"] = float((f_hemo >= thr_hemo).mean())
    thr_exud = 0.5 * np.median(r_exud)
    results["lesion_exud_presence"] = float((f_exud >= thr_exud).mean())
    results["lesion_real_hemo_median"] = float(np.median(r_hemo))
    results["lesion_real_exud_median"] = float(np.median(r_exud))

    # --- 血管 ---
    r_vf = vessel_fractions(r_rgb)
    f_vf = vessel_fractions(f_rgb)
    results["vessel_frac_wass"] = float(wasserstein_distance(r_vf, f_vf))
    results["vessel_frac_real_mean"] = float(r_vf.mean())
    results["vessel_frac_fake_mean"] = float(f_vf.mean())

    # --- 记忆检测（GPU）---
    results.update(compute_memorization(real, fake, device))

    # --- Vessel Dice（仅条件模型）---
    if cond_path:
        vd = compute_vessel_dice(fake, cond_path, device)
        results["vessel_dice"] = vd
        results["vessel_dice_note"] = "条件模型：生成图血管 vs 输入mask（mask质量有限，仅供参考）"

    # --- BRISQUE ---
    results["brisque_fake"] = compute_brisque(fake, device)
    results["brisque_real"] = compute_brisque(real, device)

    # --- C2ST ---
    if not skip_c2st:
        t0 = time.time()
        results.update(compute_c2st(real, fake, device, epochs=c2st_epochs))
        results["c2st_time_s"] = round(time.time() - t0, 1)

    if model:
        results["model"] = model
    return results


def main():
    ap = argparse.ArgumentParser(description="专用层自设计评估指标")
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cond_path", default=None, help="条件模型血管 mask 目录（仅条件模型用）")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_c2st", action="store_true")
    ap.add_argument("--c2st_epochs", type=int, default=25)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import io, contextlib
    if args.json:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results = evaluate(args.real, args.fake, args.model, args.cond_path,
                               args.img_size, args.device, args.skip_c2st, args.c2st_epochs,
                               verbose=False)
    else:
        results = evaluate(args.real, args.fake, args.model, args.cond_path,
                           args.img_size, args.device, args.skip_c2st, args.c2st_epochs)
    if results is None:
        sys.exit(1)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("=" * 56)
        print(f"  {args.model or os.path.basename(args.fake)}  专用层指标")
        print("=" * 56)
        print(f"  病灶: 出血Wass={results['lesion_hemo_wass']:.4f} 保留率={results['lesion_hemo_retention']:.3f} "
              f"存在率={results['lesion_hemo_presence']:.3f}")
        print(f"  病灶: 渗出Wass={results['lesion_exud_wass']:.4f} 保留率={results['lesion_exud_retention']:.3f} "
              f"存在率={results['lesion_exud_presence']:.3f}")
        print(f"  血管: 密度Wass={results['vessel_frac_wass']:.4f} (真实{results['vessel_frac_real_mean']:.3f} "
              f"vs 生成{results['vessel_frac_fake_mean']:.3f})")
        print(f"  记忆: 最近邻SSIM均值={results['mem_ssim_mean']:.4f} >0.85占比={results['mem_ssim_pct_gt_085']:.3f}")
        if results.get("vessel_dice") is not None:
            print(f"  Vessel Dice: {results['vessel_dice']:.4f}")
        if results.get("c2st_auc") is not None:
            print(f"  C2ST AUC: {results['c2st_auc']:.4f} (+-{results.get('c2st_auc_std', 0):.4f})")
        print(f"  BRISQUE: fake={results['brisque_fake']} real={results['brisque_real']}")


if __name__ == "__main__":
    main()
