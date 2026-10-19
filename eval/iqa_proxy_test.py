"""
A2 前置验证：无参考 IQA 代理指标 vs 作者人工排序（DL-EX-024 准备工作，零训练）
============================================================
作者"标准答案"（2026-10-19 判定，D3_ema 批 8 张）：
  sample_0 第 1（几乎可当真图）> sample_2 第 2 > sample_1 > sample_3（几乎全模糊）；
  sample_6/7 = 结构性失败（大面积亮糊）；sample_4/5 未明确排序（不参与校验）。
  跨批约束：D3_ema 批整体优于 D4_ema 批（D4 噪声多）。

候选（综述 arXiv:2501.11520 §III-B 数值型 NR-IQA + 经典代理）：
  BRISQUE(↓) / CLIP-IQA(↑) / Laplacian 锐度(↑) / 颜色统计距离(↓) / 白像素率(↓)

用法: python eval/iqa_proxy_test.py   （在仓库根目录运行）
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCHES = {
    "D3_ema": os.path.join(REPO, "generate_project/deep_learning/Fundus-Diffusion/ddpm/judgment_20261018/D3_ema/singles"),
    "D4_ema": os.path.join(REPO, "generate_project/deep_learning/Fundus-Diffusion/ddpm/judgment_20261019/D4_ema/singles"),
}
REAL_DIR = os.path.join(REPO, "fundus/combined_sq128")

# 作者排序约束（索引 a, b, a 排在 b 前）：0 第 1、2 第 2、1>3、6/7 结构性失败、4/5 在 0/2 之后
D3_ORDER = [(0, 2), (0, 1), (0, 3), (0, 6), (0, 7), (0, 4), (0, 5),
            (2, 1), (2, 3), (2, 6), (2, 7), (2, 4), (2, 5),
            (1, 3), (1, 6), (1, 7), (3, 6), (3, 7)]
D3_BAD = [6, 7]          # 结构性失败，期望排在最末
D3_TOP = [0, 2]          # 期望进前 2


def load_batch(d):
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    xs = []
    for f in files:
        img = Image.open(os.path.join(d, f)).convert("RGB").resize((128, 128))
        xs.append(torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0)
    return torch.stack(xs)


def real_color_stats():
    files = sorted(f for f in os.listdir(REAL_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    px = []
    for f in files:
        img = Image.open(os.path.join(REAL_DIR, f)).convert("RGB").resize((128, 128))
        a = np.array(img).astype(np.float32) / 255.0
        px.append(a.reshape(-1, 3))
    px = np.concatenate(px, 0)
    return px.mean(0), px.std(0)


def laplacian_var(x):
    """锐度代理：灰度图 Laplacian 响应的方差（越高越锐）。x: [B,3,H,W] in [0,1]"""
    g = x.mean(dim=1, keepdim=True)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    resp = F.conv2d(g, k, padding=1)
    return resp.var(dim=(1, 2, 3))


def color_dist(x, r_mean, r_std):
    """每图 RGB 均值/标准差与真实眼底统计的 L2 距离（越低越近）。"""
    m = x.mean(dim=(2, 3))          # [B,3]
    s = x.std(dim=(2, 3))
    return ((m - r_mean).norm(dim=1) + (s - r_std).norm(dim=1)) / 2


def white_rate(x):
    return (x > 0.95).float().mean(dim=(1, 2, 3))


def main():
    torch.manual_seed(0)
    batches = {k: load_batch(v) for k, v in BATCHES.items()}
    r_mean, r_std = torch.tensor(real_color_stats()[0]), torch.tensor(real_color_stats()[1])

    metrics = {}
    # BRISQUE（piq，↓）
    try:
        import piq
        metrics["BRISQUE↓"] = {k: piq.brisque(v, data_range=1.0, reduction="none") for k, v in batches.items()}
    except Exception as e:
        print(f"[WARN] BRISQUE 不可用: {e}")
    # CLIP-IQA（piq，↑；CLIPIQA 类，默认 prompt "Good photo." vs "Bad photo."；首次会下 CLIP 权重）
    try:
        from piq.clip_iqa import CLIPIQA
        scorer = CLIPIQA(data_range=1.0)
        metrics["CLIP-IQA↑"] = {k: scorer(v).flatten().detach().cpu() for k, v in batches.items()}
    except Exception as e:
        print(f"[WARN] CLIP-IQA 不可用: {e}")
    # 经典代理
    metrics["锐度↑"] = {k: laplacian_var(v) for k, v in batches.items()}
    metrics["颜色距离↓"] = {k: color_dist(v, r_mean, r_std) for k, v in batches.items()}
    metrics["白像素率↓"] = {k: white_rate(v) for k, v in batches.items()}

    # ---- 输出表 ----
    d3 = batches["D3_ema"]
    n3 = d3.shape[0]
    print(f"\n{'样本':>4} | " + " | ".join(f"{k}" for k in metrics))
    for i in range(n3):
        print(f"  D3-{i} | " + " | ".join(f"{metrics[k]['D3_ema'][i].item():8.3f}" for k in metrics))
    for i in range(batches["D4_ema"].shape[0]):
        print(f"  D4-{i} | " + " | ".join(f"{metrics[k]['D4_ema'][i].item():8.3f}" for k in metrics))

    # ---- 校验 1：作者排序约束（D3 批，仅已知对）----
    print("\n=== 校验 1：作者排序约束（0>2>其余；1>3；6/7 垫底；共 18 对）===")
    print(f"{'指标':>10} | 违反对数 /18 | top2 命中 | 6/7 是否垫底")
    for name, per in metrics.items():
        v = per["D3_ema"]
        higher_better = name.endswith("↑")
        s = v if higher_better else -v  # 统一成"越大越好"
        viol = sum(1 for a, b in D3_ORDER if not (s[a] > s[b]))
        order = torch.argsort(s, descending=True).tolist()  # 最好在前
        top2 = sorted(order[:2]) == sorted(D3_TOP)
        bottom2 = set(order[-2:]) >= set(D3_BAD)
        print(f"{name:>10} | {viol:>4} /18     | {top2!s:^9} | {bottom2!s:^8}  (最好→最差: {order})")

    # ---- 校验 2：跨批区分（D3 批应整体优于 D4 批）----
    print("\n=== 校验 2：跨批区分（D3 批整体应优于 D4 批）===")
    for name, per in metrics.items():
        higher_better = name.endswith("↑")
        m3, m4 = per["D3_ema"].mean().item(), per["D4_ema"].mean().item()
        ok = (m3 > m4) if higher_better else (m3 < m4)
        print(f"{name:>10} | D3均值 {m3:8.3f} vs D4均值 {m4:8.3f} → {'✅ 方向正确' if ok else '❌ 方向错误'}")


if __name__ == "__main__":
    main()
