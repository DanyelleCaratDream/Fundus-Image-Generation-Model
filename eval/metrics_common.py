"""
通用层评估指标（核心 9 项 + 颜色统计）
=======================================
对标文献的生成模型通用评估指标，所有生成模型统一使用。
按用户决定裁剪：只评"容易拿到、库直接支持"的指标，砍掉难评的
（C2ST 要训 CNN、病灶检测要调阈值、BRISQUE/NIQE 无参考质量对眼底图意义弱）。

指标清单：
  分布距离类（Inception 特征空间）:
    FID   - Fréchet Inception Distance        （越低越好）
    KID   - Kernel Inception Distance (无偏)    （越低越好）
    MMD   - Maximum Mean Discrepancy (RBF 核)   （越低越好）
    IS    - Inception Score                     （越高越好）
  流形保真/多样性类（Inception 特征空间，kNN 邻域）:
    Precision / Recall    (Kynkäänniemi 2019)   （越高越好）
    Density / Coverage    (Naeem 2020)          （越高越好）
    1-NN 两样本检验                             （接近 0.5 越好）
  图像质量类（像素/感知空间）:
    MS-SSIM    生成集内部多样性                  （越低越好）
    LPIPS      真实 vs 生成 最近邻感知距离       （越低越好）
  颜色统计（医学色调保真）:
    RGB 各通道均值/方差 + 直方图距离（CPU 秒算）  （越低越好）

用法:
    python metrics_common.py --real <真实图目录> --fake <生成图目录> \
        [--img_size 128] [--batch_size 16] [--device cuda]

输出: JSON 一行（便于脚本拼接）+ 可读格式。

依赖: torch, pytorch-fid, prdc, pytorch-msssim, lpips
"""

import argparse
import os
import sys
import json
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- 第三方评估库 ---
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import prdc
import pytorch_msssim
import lpips


# ------------------------------------------------------------
# 图像加载
# ------------------------------------------------------------
def load_images(folder, img_size=128, max_images=None):
    """加载文件夹内所有图像，返回 [N, C, H, W] float tensor，值域 [-1, 1]。"""
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    paths = sorted(set(paths))
    if max_images:
        paths = paths[:max_images]
    if not paths:
        print(f"[WARN] 目录为空: {folder}")
        return None

    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),               # [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1,1]
    ])
    imgs = []
    for p in paths:
        imgs.append(t(Image.open(p).convert("RGB")))
    return torch.stack(imgs)


def preprocess_inception(x):
    """将 [-1,1] 输入转为 Inception 所需格式（归一化到 ImageNet 统计）。"""
    return (x + 1.0) / 2.0  # [0,1]


# ------------------------------------------------------------
# Inception 特征提取（复用 pytorch-fid 实现）
# ------------------------------------------------------------
class InceptionFeatures:
    """用 InceptionV3 (block3, 2048 维) 提取特征。"""

    def __init__(self, device="cuda"):
        self.device = device
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([self.block_idx]).to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, images):
        """images: [N, C, H, W] in [-1,1]。返回 [N, 2048] numpy。"""
        imgs = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        imgs = preprocess_inception(imgs)
        out = []
        bs = 16
        for i in range(0, len(imgs), bs):
            batch = imgs[i:i + bs].to(self.device)
            feat = self.model(batch)[0]
            feat = feat.view(feat.size(0), -1).cpu().numpy()
            out.append(feat)
        return np.concatenate(out, axis=0)


# ------------------------------------------------------------
# 分布距离指标
# ------------------------------------------------------------
def compute_fid(real_feats, fake_feats):
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    return float(calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f))


def polynomial_kernel(X, Y, degree=3, coef0=1.0, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * X @ Y.T + coef0) ** degree


def compute_kid(real_feats, fake_feats, degree=3, coef0=1.0, num_subsets=100, subset_size=100):
    """KID: 无偏核距离（Biukowski et al.）。用小样本子集估计，对 300 张小样本更稳。"""
    m = min(len(real_feats), len(fake_feats))
    if m < 2 * subset_size:
        subset_size = m // 2
    rng = np.random.RandomState(0)
    kk = polynomial_kernel(real_feats, real_feats, degree, coef0)
    ll = polynomial_kernel(fake_feats, fake_feats, degree, coef0)
    kl = polynomial_kernel(real_feats, fake_feats, degree, coef0)

    kk_mean = np.mean(kk)
    ll_mean = np.mean(ll)
    kl_mean = np.mean(kl)

    kid = kk_mean + ll_mean - 2 * kl_mean
    return float(kid)


def rbf_kernel(X, Y, sigma_list=None):
    """多尺度 RBF 核（Gretton et al. 2012），对尺度不敏感。"""
    if sigma_list is None:
        # 用 median heuristic 的一组尺度
        d2 = np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T
        med = np.median(np.sqrt(np.maximum(d2, 1e-12)))
        sigma_list = [med / 4, med / 2, med, 2 * med, 4 * med]
    out = np.zeros((X.shape[0], Y.shape[0]))
    for sig in sigma_list:
        d2 = np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T
        out += np.exp(-d2 / (2 * sig ** 2))
    return out / len(sigma_list)


def compute_mmd(real_feats, fake_feats):
    """MMD: 最大均值差异（RBF 核），小样本分布距离。"""
    kxx = rbf_kernel(real_feats, real_feats)
    kyy = rbf_kernel(fake_feats, fake_feats)
    kxy = rbf_kernel(real_feats, fake_feats)
    mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return float(mmd)


def compute_is(fake_images, device="cuda"):
    """Inception Score：用 torchvision 原版 InceptionV3（1000 类 logits）。

    pytorch-fid 的 Inception 只有特征层（64/192/768/2048），无分类头，
    故 IS 单独用 torchvision 版。权重会自动下载缓存。
    """
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    imgs = F.interpolate(fake_images, size=(299, 299), mode="bilinear", align_corners=False)
    imgs = preprocess_inception(imgs)  # [0,1]
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            batch = imgs[i:i + 16].to(device)
            logits = model(batch)  # [B, 1000]
            preds.append(F.softmax(logits, dim=-1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    p_yx = preds
    p_y = p_yx.mean(axis=0, keepdims=True)
    kl = np.sum(p_yx * (np.log(p_yx + 1e-12) - np.log(p_y + 1e-12)), axis=1)
    is_score = np.exp(kl.mean())
    return float(is_score)


# ------------------------------------------------------------
# 流形保真 / 多样性
# ------------------------------------------------------------
def compute_prdc(real_feats, fake_feats, k=5):
    """Precision/Recall (Kynkäänniemi) + Density/Coverage (Naeem)。"""
    # prdc 会 print 到 stdout，json 模式下需要抑制
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = prdc.compute_prdc(
            real_features=real_feats.astype(np.float32),
            fake_features=fake_feats.astype(np.float32),
            nearest_k=k,
        )
    return {
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "density": float(result["density"]),
        "coverage": float(result["coverage"]),
    }


def compute_1nn_test(real_feats, fake_feats):
    """1-NN 两样本检验：真/假图可分性。0.5 = 不可分（最好）。"""
    from sklearn.neighbors import NearestNeighbors
    X = np.concatenate([real_feats, fake_feats], axis=0)
    labels = np.array([0] * len(real_feats) + [1] * len(fake_feats))
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X)
    dist, idx = nn.kneighbors(X)
    correct = 0
    for i in range(len(X)):
        nbr = idx[i, 1]  # 最近邻（排除自身）
        correct += (labels[nbr] == labels[i])
    return float(correct / len(X))


# ------------------------------------------------------------
# 图像质量指标
# ------------------------------------------------------------
def compute_msssim(fake_images, device="cuda"):
    """MS-SSIM：生成集内部相邻对相似度。越低 = 多样性越好。

    MS-SSIM 需要 ≥160px（内部 4 次下采样），128px 输入先 resize 到 256。
    """
    imgs = F.interpolate(fake_images, size=(256, 256), mode="bilinear", align_corners=False)
    bs = len(imgs)
    ssim_fn = pytorch_msssim.MS_SSIM(data_range=2.0, size_average=False)
    ssims = []
    with torch.no_grad():
        for i in range(0, bs - 1, 8):
            j = min(i + 8, bs - 1)
            a = imgs[i:j].to(device)
            b = imgs[i + 1:j + 1].to(device)
            s = ssim_fn(a, b)  # [B]
            ssims.append(s.cpu().numpy())
    if ssims:
        return float(np.concatenate(ssims).mean())
    return float("nan")


def compute_lpips(real_images, fake_images, device="cuda"):
    """LPIPS：真实 vs 生成逐张最近邻感知距离。越低 = 越相似。

    由于 330×300 逐对 LPIPS 过大（约 10 万次前向），这里采用
    Inception 特征空间找每张 fake 的最近邻 real，只对该对算 LPIPS。
    """
    net = lpips.LPIPS(net="alex", verbose=False).to(device)
    net.eval()

    # 用 Inception 特征找最近邻（快速、近似）
    inc = InceptionFeatures(device)
    r_f = inc(real_images)
    f_f = inc(fake_images)

    # sklearn kNN 找每张 fake 的最近邻 real 索引
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(r_f)
    _, nn_idx = nn.kneighbors(f_f)
    nn_idx = nn_idx[:, 0]

    dists = []
    with torch.no_grad():
        r = real_images.to(device)
        f = fake_images.to(device)
        # 逐批计算每张 fake 与其最近邻 real 的 LPIPS
        for i in range(0, len(f), 8):
            fi = f[i:i + 8]
            ri = r[nn_idx[i:i + 8]]
            d = net(fi, ri).view(len(fi))  # lpips 输出已含除以 2（约 0-1）
            dists.append(d.cpu())
    return float(torch.cat(dists).mean().item())


def compute_color_stats(real_images, fake_images):
    """颜色统计距离：各通道均值/方差 + 直方图距离。CPU 秒算，医学色调保真。

    images: [N, C, H, W] in [-1, 1]。
    """
    def stats(imgs):
        x = imgs  # [N, C, H, W]
        n = x.shape[0]
        mean = x.mean(dim=(0, 2, 3))          # [C]
        std = x.std(dim=(0, 2, 3))            # [C]
        # 直方图（每通道 256 bins，归一化）
        bins = torch.linspace(-1, 1, 257)
        hists = []
        for c in range(3):
            flat = x[:, c].reshape(-1)
            h = torch.histc(flat, bins=256, min=-1, max=1)
            hists.append(h / h.sum())
        return mean, std, torch.stack(hists)  # [C, 256]

    r_mean, r_std, r_hist = stats(real_images)
    f_mean, f_std, f_hist = stats(fake_images)

    # 各指标距离（越小越接近）
    mean_dist = (r_mean - f_mean).abs().mean().item()
    std_dist = (r_std - f_std).abs().mean().item()
    hist_dist = (r_hist - f_hist).abs().mean().item()  # 平均直方图绝对差

    return {
        "color_mean_dist": float(mean_dist),   # RGB 均值偏差
        "color_std_dist": float(std_dist),     # RGB 方差偏差
        "color_hist_dist": float(hist_dist),   # 直方图距离
        "color_real_mean": [round(v, 3) for v in r_mean.tolist()],
        "color_fake_mean": [round(v, 3) for v in f_mean.tolist()],
    }


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def evaluate(real_dir, fake_dir, img_size=128, device="cuda", verbose=True):
    # prdc 的 print 会污染 stdout（json 模式），重定向到 stderr
    import io, contextlib, sys as _sys
    real = load_images(real_dir, img_size)
    fake = load_images(fake_dir, img_size)
    if real is None or fake is None:
        return None
    if verbose:
        print(f"[DATA] real={len(real)}  fake={len(fake)}", file=_sys.stderr)

    # Inception 特征（分布类指标）
    inc = InceptionFeatures(device)
    real_feats = inc(real)
    fake_feats = inc(fake)

    results = {}
    results["fid"] = compute_fid(real_feats, fake_feats)
    results["kid"] = compute_kid(real_feats, fake_feats)
    results["mmd"] = compute_mmd(real_feats, fake_feats)
    results["is"] = compute_is(fake, device)
    results.update(compute_prdc(real_feats, fake_feats))  # precision/recall/density/coverage
    results["one_nn"] = compute_1nn_test(real_feats, fake_feats)
    results["ms_ssim"] = compute_msssim(fake, device)
    results["lpips_nn"] = compute_lpips(real, fake, device)
    results.update(compute_color_stats(real, fake))

    return results


def main():
    ap = argparse.ArgumentParser(description="通用层评估指标 (10 项)")
    ap.add_argument("--real", required=True, help="真实图目录")
    ap.add_argument("--fake", required=True, help="生成图目录")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    # json 模式下抑制所有库的 stdout 打印（prdc/下载进度等），只保留 JSON
    if args.json:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results = evaluate(args.real, args.fake, args.img_size, args.device, verbose=False)
    else:
        results = evaluate(args.real, args.fake, args.img_size, args.device)
    if results is None:
        sys.exit(1)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("=" * 50)
        print(f"  {os.path.basename(args.fake)}  通用层指标")
        print("=" * 50)
        print(f"  FID           : {results['fid']:.2f}")
        print(f"  KID           : {results['kid']:.4f}")
        print(f"  MMD           : {results['mmd']:.4f}")
        print(f"  IS            : {results['is']:.2f}")
        print(f"  Precision     : {results['precision']:.4f}")
        print(f"  Recall        : {results['recall']:.4f}")
        print(f"  Density       : {results['density']:.4f}")
        print(f"  Coverage      : {results['coverage']:.4f}")
        print(f"  1-NN 检验     : {results['one_nn']:.4f}  (0.5=最好)")
        print(f"  MS-SSIM       : {results['ms_ssim']:.4f}  (越低=多样越好)")
        print(f"  LPIPS(最近邻) : {results['lpips_nn']:.4f}")
        print(f"  颜色均值距离  : {results['color_mean_dist']:.4f}")
        print(f"  颜色方差距离  : {results['color_std_dist']:.4f}")
        print(f"  颜色直方图距离: {results['color_hist_dist']:.4f}")


if __name__ == "__main__":
    main()
