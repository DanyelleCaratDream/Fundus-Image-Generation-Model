# -*- coding: utf-8 -*-
"""
eval/plot_metrics.py —— 评估指标可视化（柱状图 + 雷达图）

用途：
  把 eval_data/ 下的各模型评估结果画成图，整合进评估报告（research-report/evaluation_report.md）。

可复用：
  脚本自动扫描 --datadir 下的 *_metrics.json（通用层）和 *_fundus_metrics.json（专用层）。
  以后 Phase C2（传统 ML 方法）等新方法跑完评估，只要把结果 JSON 放进 eval_data/，
  重跑本脚本，新模型会自动追加进所有图表（模型显示名可在 MODEL_LABELS 里补充）。

用法（在项目根目录）：
  python eval/plot_metrics.py                       # 输出到 research-report/figures/
  python eval/plot_metrics.py --outdir <dir>        # 自定义输出目录
  python eval/plot_metrics.py --datadir <dir>       # 自定义数据目录（默认 eval_data/）

输出：
  figures/general_metrics.png   通用层 11 项指标 + 人工评分（多面板柱状图）
  figures/color_metrics.png     颜色统计 3 项（多面板柱状图）
  figures/fundus_metrics.png    专用层自设计指标（病灶/血管/记忆/C2ST/BRISQUE）
  figures/radar.png             关键指标归一化雷达图（跨两层）
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务器/脚本安全
import matplotlib.pyplot as plt
import numpy as np

# ---- 中文字体（Windows） ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 模型元信息（显示名 + 配色 + 人工评分）
# 新模型键不在下面时：显示名退回键名、配色按顺序复用、人工评分计为 None
# ============================================================
MODEL_LABELS = {
    "film_l1lpips": "FiLM+L1+LPIPS",
    "film": "FiLM MSE",
    "cond": "条件扩散",
    "base_cj": "基础DDPM",
    "dcgan": "DCGAN",
    "vae": "VAE Large",
}
MODEL_COLORS = {
    "film_l1lpips": "#c0392b",  # 红（最佳，突出）
    "film": "#e67e22",          # 橙
    "cond": "#2980b9",          # 蓝
    "base_cj": "#16a085",       # 绿
    "dcgan": "#8e44ad",         # 紫
    "vae": "#7f8c8d",           # 灰
}
# 人工评分（报告 5.4 综合排名；85-90 取中间值 87.5 画图，备注在报告里写清）
HUMAN_SCORE = {
    "film_l1lpips": 87.5,
    "film": 85,
    "base_cj": 75,
    "cond": 70,
    "dcgan": 20,
    "vae": 10,
}
# 模型展示顺序：排在前的画在左；不在列表里的新模型自动追加到末尾
MODEL_ORDER = ["film_l1lpips", "film", "cond", "base_cj", "dcgan", "vae"]

# 通用层面板定义: (json 键, 中文标签, 方向, 数值格式, 是否 log 轴)
GENERAL_PANELS = [
    ("fid", "FID ↓", "lower", ".1f", False),
    ("kid", "KID ↓", "lower", ".3f", False),
    ("mmd", "MMD ↓", "lower", ".4f", False),
    ("is", "IS ↑", "higher", ".2f", False),
    ("precision", "Precision ↑", "higher", ".3f", False),
    ("recall", "Recall ↑", "higher", ".3f", False),
    ("density", "Density ↑", "higher", ".3f", False),
    ("coverage", "Coverage ↑", "higher", ".3f", False),
    ("one_nn", "1-NN (→0.5)", "close", ".3f", False),
    ("ms_ssim", "MS-SSIM ↓", "lower", ".3f", False),
    ("lpips_nn", "LPIPS ↓", "lower", ".3f", False),
]

# 颜色统计面板
COLOR_PANELS = [
    ("color_mean_dist", "均值距离 ↓", ".3f", False),
    ("color_std_dist", "方差距离 ↓", ".3f", False),
    ("color_hist_dist", "直方图距离 ↓", ".4f", False),
]

# 专用层面板（键名来自 *_fundus_metrics.json）
FUNDUS_PANELS = [
    ("lesion_hemo_wass", "出血 Wass ↓", ".3f", False),
    ("lesion_hemo_retention", "出血保留率 (≈1)", ".2f", False),
    ("lesion_exud_wass", "渗出 Wass ↓", ".3f", False),
    ("lesion_exud_retention", "渗出保留率", ".1f", True),   # 量级跨度大，log 轴
    ("vessel_frac_wass", "血管占比 Wass ↓", ".3f", False),
    ("vessel_dice", "Vessel Dice ↑", ".3f", False),
    ("mem_ssim_mean", "记忆 NN-SSIM ↓", ".3f", False),
    ("c2st_auc", "C2ST AUC ↓", ".3f", False),
    ("brisque_fake", "BRISQUE ↓", ".1f", False),
]

# 雷达图关键指标（跨两层，方向归一化；元组: 显示名, (json 文件类型, 键名 或 "human"), 方向）
#   文件类型: "g" = 通用层 _metrics.json；"f" = 专用层 _fundus_metrics.json；"h" = 人工评分
RADAR_METRICS = [
    ("FID", ("g", "fid"), "lower"),
    ("KID", ("g", "kid"), "lower"),
    ("IS", ("g", "is"), "higher"),
    ("Recall", ("g", "recall"), "higher"),
    ("Coverage", ("g", "coverage"), "higher"),
    ("MS-SSIM", ("g", "ms_ssim"), "lower"),
    ("LPIPS", ("g", "lpips_nn"), "lower"),
    ("C2ST", ("f", "c2st_auc"), "lower"),
    ("人工分", ("h", None), "higher"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_models(datadir):
    """扫描通用层 *_metrics.json（排除 *_fundus_metrics.json），返回 (model_key, metrics_dict) 列表。"""
    models = {}
    for p in sorted(glob.glob(os.path.join(datadir, "*_metrics.json"))):
        base = os.path.basename(p)
        if base.endswith("_fundus_metrics.json"):
            continue
        key = base[: -len("_metrics.json")]
        models[key] = load_json(p)

    # 兼容无通用层文件但只有专用层文件的模型（如未来某方法只跑专用层）
    for p in sorted(glob.glob(os.path.join(datadir, "*_fundus_metrics.json"))):
        key = os.path.basename(p)[: -len("_fundus_metrics.json")]
        if key not in models:
            models[key] = {}
    return models


def load_fundus(datadir, model_key):
    p = os.path.join(datadir, f"{model_key}_fundus_metrics.json")
    if os.path.exists(p):
        return load_json(p)
    return {}


def model_order(models):
    """返回展示顺序：MODEL_ORDER 里存在的先排，新模型按字母追加。"""
    keys = list(models.keys())
    ordered = [k for k in MODEL_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in MODEL_ORDER)
    return ordered + rest


def fmt_label(ax, vals, fmt, direction=None):
    """给柱状图柱子顶部加数值标签。"""
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.text(i, v, f"{v:{fmt}}", ha="center", va="bottom", fontsize=7)


def bar_panel(ax, models, labels, colors, panel):
    """画一个指标的多模型柱状图。panel = (json_key, label, direction, fmt, log)。"""
    key, label, direction, fmt, use_log = panel
    vals = [models[m].get(key) for m in models]  # None = 该模型无此指标
    xs = np.arange(len(models))
    ax.bar(xs, [v if v is not None else 0 for v in vals],
           color=[colors.get(m, "#95a5a6") for m in models], width=0.65)
    if use_log:
        ax.set_yscale("log")
    fmt_label(ax, vals, fmt, direction)
    ax.set_title(label, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[m] for m in models], rotation=30, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)


def draw_general(axs, models, labels, colors, human):
    """通用层 11 项 + 人工评分（12 个面板，3 行 × 4 列）。"""
    panels = list(GENERAL_PANELS)
    # 人工评分不在 JSON 里，单独补一个面板
    human_vals = [human.get(m) for m in models]
    for ax, (key, label, direction, fmt, use_log) in zip(axs.flat, panels + [(None, "人工评分 ↑", "higher", ".0f", False)]):
        if key is None:
            vals = human_vals
            ax.bar(np.arange(len(models)), [v if v is not None else 0 for v in vals],
                   color=[colors.get(m, "#95a5a6") for m in models], width=0.65)
            fmt_label(ax, vals, ".0f")
        else:
            bar_panel(ax, models, labels, colors, (key, label, direction, fmt, use_log))


def draw_color(axs, models, labels, colors):
    for ax, panel in zip(axs.flat, COLOR_PANELS):
        key, label, fmt, use_log = panel
        vals = [models[m].get(key) for m in models]
        ax.bar(np.arange(len(models)), [v if v is not None else 0 for v in vals],
               color=[colors.get(m, "#95a5a6") for m in models], width=0.65)
        fmt_label(ax, vals, fmt)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([labels[m] for m in models], rotation=30, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)


def draw_fundus(axs, models, labels, colors, fundus_map):
    for ax, panel in zip(axs.flat, FUNDUS_PANELS):
        key, label, fmt, use_log = panel
        vals = [fundus_map[m].get(key) for m in models]
        ax.bar(np.arange(len(models)), [v if v is not None else 0 for v in vals],
               color=[colors.get(m, "#95a5a6") for m in models], width=0.65)
        if use_log:
            ax.set_yscale("log")
        fmt_label(ax, vals, fmt)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([labels[m] for m in models], rotation=30, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)


def draw_score_overview(scores_data, models, labels, colors, human, outpath, dpi):
    """综合评分总览（0-100 色带 + 人工分对照），读 eval_data/_scores.json（score_scheme.py 生成）。"""
    scores = scores_data.get("scores")
    if not scores:
        print("  [score] _scores.json 里没有 scores，跳过综合评分面板")
        return
    order = sorted(scores, key=scores.get, reverse=True)
    vals = [scores[m] for m in order]
    xs = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=dpi)
    bands = [(0, 30, "#e74c3c", "失败带"), (30, 50, "#f39c12", "较差"),
             (50, 70, "#f1c40f", "一般"), (70, 85, "#27ae60", "良好"), (85, 100, "#16a085", "优秀")]
    for lo, hi, c, name in bands:
        ax.axhspan(lo, hi, color=c, alpha=0.08)
        if lo == 0:
            ax.text(len(order) - 0.35, hi + 0.4, name, fontsize=7, color="#666",
                    ha="right", va="bottom")
    ax.bar(xs, vals, color=[colors.get(m, "#95a5a6") for m in order], width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=9, zorder=4)
    hv = [human.get(m) for m in order]
    ax.plot(xs, hv, "k--o", markersize=4, alpha=0.5, label="人工分", zorder=5)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[m] for m in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("综合评分（0-100）")
    ax.set_title("综合评分总览（公式 " + str(scores_data.get("formula_version", "?")) +
                 "，门控: " + str(scores_data.get("gate", "?")) + "）", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [score] 写入 {outpath}")


def draw_radar(models, labels, colors, human, fundus_map, outpath, dpi):
    """归一化雷达图。每轴：方向反转后 min-max 归一化到 [0,1]，1 = 组内最佳。"""
    n = len(RADAR_METRICS)
    all_vals = {}
    for name, (ftype, key), direction in RADAR_METRICS:
        vals = {}
        for m in models:
            if ftype == "h":
                v = human.get(m)
            elif ftype == "g":
                v = models[m].get(key)
            else:
                v = fundus_map[m].get(key)
            vals[m] = v
        all_vals[name] = vals

    # 归一化到 [0,1]
    norm = {}
    for name, (ftype, key), direction in RADAR_METRICS:
        raw = {m: v for m, v in all_vals[name].items() if v is not None}
        if not raw:
            norm[name] = {m: None for m in models}
            continue
        lo, hi = min(raw.values()), max(raw.values())
        rng = hi - lo if hi > lo else 1.0
        norm[name] = {}
        for m in models:
            v = raw.get(m)
            if v is None:
                norm[name][m] = None
            elif direction == "higher":
                norm[name][m] = (v - lo) / rng
            elif direction == "close":   # 越接近 0.5 越好（未用于雷达，保留兜底）
                norm[name][m] = 1 - abs(v - 0.5) / 0.5
            else:
                norm[name][m] = (hi - v) / rng

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5), dpi=dpi)
    ax = fig.add_subplot(111, polar=True)
    labels_list = [m for m in models if all(norm[nm].get(m) is not None for nm, *_ in RADAR_METRICS)]
    if not labels_list:
        print("[radar] 没有模型拥有全部雷达指标，跳过雷达图")
        plt.close(fig)
        return
    for m in labels_list:
        vals = [norm[nm].get(m, None) for nm, *_ in RADAR_METRICS]
        if any(v is None for v in vals):
            continue
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=labels[m], color=colors.get(m, "#95a5a6"))
        ax.fill(angles, vals, alpha=0.06, color=colors.get(m, "#95a5a6"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([nm for nm, *_ in RADAR_METRICS], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=7, color="#999999")
    ax.set_title("关键指标归一化对比（1 = 组内最佳；方向已反转为“越大越好”）", fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [radar] 写入 {outpath}")


def main():
    ap = argparse.ArgumentParser(description="评估指标可视化（柱状图 + 雷达图）")
    ap.add_argument("--datadir", default="eval_data", help="评估 JSON 目录（默认 eval_data/）")
    ap.add_argument("--outdir", default="research-report/figures", help="图表输出目录")
    ap.add_argument("--dpi", type=int, default=150, help="输出分辨率（默认 150）")
    args = ap.parse_args()

    models = discover_models(args.datadir)
    if not models:
        print(f"[错误] {args.datadir} 下没有找到 *_metrics.json")
        sys.exit(1)

    order = model_order(models)
    labels = {m: MODEL_LABELS.get(m, m) for m in order}
    colors = {m: MODEL_COLORS.get(m, None) for m in order}
    # 未配置颜色的新模型：按默认色板顺序分配
    fallback = ["#34495e", "#f39c12", "#9b59b6", "#1abc9c", "#e74c3c", "#3498db", "#2ecc71"]
    fi = 0
    for m in order:
        if colors[m] is None:
            colors[m] = fallback[fi % len(fallback)]
            fi += 1

    fundus_map = {m: load_fundus(args.datadir, m) for m in order}

    os.makedirs(args.outdir, exist_ok=True)
    print(f"发现 {len(order)} 个模型: {', '.join(order)}")
    print(f"输出目录: {os.path.abspath(args.outdir)}\n")

    # ---- 图1：通用层（3×4 = 12 面板：11 项 + 人工评分） ----
    fig, axs = plt.subplots(3, 4, figsize=(16, 9), dpi=args.dpi)
    draw_general(axs, {m: models[m] for m in order}, labels, colors, HUMAN_SCORE)
    fig.suptitle("通用层指标对比（↓ 越小越好，↑ 越大越好；1-NN 越接近 0.5 越好）", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.outdir, "general_metrics.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [general] 写入 {out}")

    # ---- 图2：颜色统计（1×3） ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), dpi=args.dpi)
    draw_color(axs, {m: models[m] for m in order}, labels, colors)
    fig.suptitle("颜色统计（真实 RGB 均值 [-0.031, -0.423, -0.707]）", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(args.outdir, "color_metrics.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [color] 写入 {out}")

    # ---- 图3：专用层自设计（3×3 = 9 面板） ----
    fig, axs = plt.subplots(3, 3, figsize=(16, 9), dpi=args.dpi)
    draw_fundus(axs, {m: models[m] for m in order}, labels, colors, fundus_map)
    fig.suptitle("专用层自设计指标（病灶/血管/记忆/C2ST/BRISQUE）", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.outdir, "fundus_metrics.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fundus] 写入 {out}")

    # ---- 图4：雷达图 ----
    draw_radar({m: models[m] for m in order}, labels, colors, HUMAN_SCORE,
               fundus_map, os.path.join(args.outdir, "radar.png"), args.dpi)

    # ---- 图5：综合评分总览（可选，读 _scores.json；无则优雅跳过） ----
    sp = os.path.join(args.datadir, "_scores.json")
    if os.path.exists(sp):
        try:
            scores_data = load_json(sp)
        except Exception as e:
            print(f"  [score] _scores.json 读取失败，跳过综合评分面板: {e}")
        else:
            draw_score_overview(scores_data, {m: models[m] for m in order}, labels, colors,
                                HUMAN_SCORE, os.path.join(args.outdir, "score_overview.png"), args.dpi)
    else:
        print("  [score] 未找到 _scores.json（先运行 eval/score_scheme.py），跳过综合评分面板")

    print("\n完成。图表已写入 evaluation_report.md 第三部分引用目录。")


if __name__ == "__main__":
    main()
