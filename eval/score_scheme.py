# -*- coding: utf-8 -*-
"""
eval/score_scheme.py —— 眼底图生成模型「综合评分方案」

用途：
  两层 ~30 个指标测完没有综合分，本脚本把指标整合成六维加权 + 现实主义门控的
  纯自动 0-100 总分，并用人工评分（HUMAN_SCORE）校准各指标：
  - 判断哪些指标贴合人眼（保留进计分）、哪些有偏（去除/降权/仅参考）
  - 新方法（含传统 ML）跑完评估，把结果 JSON 放进 eval_data/ 重跑本脚本即可客观打分

复用 eval/plot_metrics.py 的模型元信息（MODEL_LABELS/COLORS/HUMAN_SCORE/ORDER），
零重复维护；plot_metrics 的 main() 有 __name__ 守卫，import 安全。

用法（在项目根目录）：
  python eval/score_scheme.py                       # 扫 eval_data/ → 打印校准表+评分表 → 写 _scores.json
  python eval/score_scheme.py --datadir <dir>       # 自定义数据目录
  python eval/score_scheme.py --models film_l1lpips,newml  # 只看子集
  python eval/score_scheme.py --no-gate             # 对照输出未门控"原始分"
  python eval/score_scheme.py --scorecard           # 画 scorecard.png 到 --outdir（默认 research-report/figures/）

输出：
  eval_data/_scores.json   综合评分（含 formula_version/weights/gate/tau）
  figures/scorecard.png    总分条形（0-100 色带）+ 六维雷达（--scorecard 时）
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_metrics import (MODEL_LABELS, MODEL_COLORS, HUMAN_SCORE, MODEL_ORDER,
                          discover_models, load_fundus, model_order)

# 中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FORMULA_VERSION = "v1"
FORMULA_GATE = "D2 抗识破+分布"  # 现实主义门控维度

# ============================================================
# 评分公式本体（报告 5.7 逐字引用，改权重必须同步报告 + docs/09）
# dims: (中文名, 权重, 是否门控, [(文件类型, JSON键, 方向, 维内权重), ...])
#   文件类型: "g"=通用层 _metrics.json， "f"=专用层 _fundus_metrics.json
#   方向: "lower" 越小越好 / "higher" 越大越好 / "close1" 越接近 1 越好 / "close05" 越接近 0.5 越好
#
# 六维设计动机（完整"为什么"见 docs/09-Score-Scheme-Design.md）：
#   D1 病灶保留 0.30 —— KW-IV 任务核心（出血/渗出是分类器识别"最重级"的关键信号），权重最高
#   D2 抗识破+分布 0.25 —— "像不像真实"；C2ST ρ=-0.771/KID ρ=-0.886 实测最贴人工；门控 R=D2
#   D3 多样性/质量 0.20 —— 扩增价值（不能千篇一律）；Recall 干净分群扩散 vs GAN/VAE
#   D4 血管结构 0.10 —— 血管僵硬是人工识破主因（CCDM），但指标是低层代理 → 低权 + 门控
#   D5 颜色 0.08 —— 医学色偏有生理意义，但 VAE 全优证明可被"合法但低信息"博弈 → 低权 + 门控
#   D6 记忆风险 0.07 —— 防复制保险丝（FID 对复制不敏感），当前复制率 0% 故最小
# ============================================================
SCHEME = {
    "version": FORMULA_VERSION,
    "gate": FORMULA_GATE,
    "dims": [
        ("D1 病灶保留", 0.30, True, [
            ("f", "lesion_hemo_wass", "lower", 0.25),
            ("f", "lesion_hemo_retention", "close1", 0.30),
            ("f", "lesion_exud_wass", "lower", 0.20),
            ("f", "lesion_exud_retention", "close1", 0.25),
        ]),
        ("D2 抗识破+分布", 0.25, False, [
            ("f", "c2st_auc", "lower", 0.40),
            ("g", "fid", "lower", 0.25),
            ("g", "kid", "lower", 0.25),
            ("g", "one_nn", "close05", 0.10),
        ]),
        ("D3 多样性/质量", 0.20, False, [
            ("g", "recall", "higher", 0.40),
            ("g", "ms_ssim", "lower", 0.30),
            ("g", "is", "higher", 0.30),
        ]),
        ("D4 血管结构", 0.10, True, [
            ("f", "vessel_frac_wass", "lower", 0.50),
            ("f", "vessel_dice", "higher", 0.50),
        ]),
        ("D5 颜色", 0.08, True, [
            ("g", "color_mean_dist", "lower", 1.0 / 3),
            ("g", "color_std_dist", "lower", 1.0 / 3),
            ("g", "color_hist_dist", "lower", 1.0 / 3),
        ]),
        ("D6 记忆风险", 0.07, False, [
            ("f", "mem_ssim_mean", "lower", 1.0),
        ]),
    ],
}

DIM_WEIGHTS = {name: w for name, w, *_ in SCHEME["dims"]}
GATED_DIMS = {name for name, _w, gated, *_ in SCHEME["dims"] if gated}

# 校准候选指标（含不进计分、仅报告的参考指标）
# (文件类型, JSON键, 中文标签, 方向, 是否进计分, 所属维度/参考)
CANDIDATE_METRICS = [
    ("g", "fid", "FID", "lower", True, "D2"),
    ("g", "kid", "KID", "lower", True, "D2"),
    ("g", "mmd", "MMD", "lower", False, "参考"),
    ("g", "is", "IS", "higher", True, "D3"),
    ("g", "precision", "Precision", "higher", False, "参考"),
    ("g", "recall", "Recall", "higher", True, "D3"),
    ("g", "density", "Density", "higher", False, "参考"),
    ("g", "coverage", "Coverage", "higher", False, "参考"),
    ("g", "one_nn", "1-NN", "close05", True, "D2"),
    ("g", "ms_ssim", "MS-SSIM", "lower", True, "D3"),
    ("g", "lpips_nn", "LPIPS-NN", "lower", False, "参考"),
    ("g", "color_mean_dist", "颜色均值距离", "lower", True, "D5"),
    ("g", "color_std_dist", "颜色方差距离", "lower", True, "D5"),
    ("g", "color_hist_dist", "颜色直方图距离", "lower", True, "D5"),
    ("f", "lesion_hemo_wass", "出血 Wass", "lower", True, "D1"),
    ("f", "lesion_exud_wass", "渗出 Wass", "lower", True, "D1"),
    ("f", "lesion_hemo_retention", "出血保留率", "close1", True, "D1"),
    ("f", "lesion_exud_retention", "渗出保留率", "close1", True, "D1"),
    ("f", "vessel_frac_wass", "血管占比 Wass", "lower", True, "D4"),
    ("f", "vessel_dice", "Vessel Dice", "higher", True, "D4"),
    ("f", "mem_ssim_mean", "记忆 NN-SSIM", "lower", True, "D6"),
    ("f", "brisque_fake", "BRISQUE", "lower", False, "参考"),
    ("f", "c2st_auc", "C2ST AUC", "lower", True, "D2"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _r4(v):
    """四舍五入到 4 位；None 原样返回（缺失维写入 JSON 用 null，不用 NaN）。"""
    return None if v is None else round(v, 4)


def merged_metrics(datadir, model_key):
    """合并通用层 + 专用层指标为单个 dict。"""
    m = {}
    gp = os.path.join(datadir, f"{model_key}_metrics.json")
    if os.path.exists(gp):
        m.update(load_json(gp))
    m.update(load_fundus(datadir, model_key))
    return m


# ============================================================
# 归一化（min-max [0,1]，方向反转；缺失忽略，仅对存在的值归一化）
# ============================================================
def norm_lower(vals):
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1.0
    return {v: (hi - v) / rng for v in vals}


def norm_higher(vals):
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi > lo else 1.0
    return {v: (v - lo) / rng for v in vals}


def norm_close1(vals):
    """保留率：越接近 1 越好；scale 取 max(1, max|v-1|) 防被大离群值压扁。"""
    scale = max(1.0, max(abs(v - 1.0) for v in vals))
    return {v: max(0.0, 1.0 - abs(v - 1.0) / scale) for v in vals}


def norm_close05(vals):
    """1-NN：越接近 0.5 越好；全模型都接近 1 时贡献≈0（饱和，降权效果）。"""
    return {v: max(0.0, 1.0 - abs(v - 0.5) / 0.5) for v in vals}


def normalize(vals, direction):
    if direction == "lower":
        return norm_lower(vals)
    if direction == "higher":
        return norm_higher(vals)
    if direction == "close1":
        return norm_close1(vals)
    if direction == "close05":
        return norm_close05(vals)
    raise ValueError(f"未知方向: {direction}")


# ============================================================
# 六维评分 + 门控 + 缺失键容错
# ============================================================
def dim_scores(all_metrics, dim):
    """计算各模型在某维度的得分 dict；整维缺失的模型不在返回里。"""
    name, w, _gated, members = dim
    # 每个成员指标先跨模型归一化，再取各模型的值
    per_member = {}   # member -> {model: norm_value}
    for ftype, key, direction, mw in members:
        raw = {}
        for m, md in all_metrics.items():
            v = md.get(key)
            if v is not None:
                raw[m] = v
        if raw:
            norm_map = normalize(list(raw.values()), direction)
            per_member[key] = {m: norm_map[raw[m]] for m in raw}

    # 每个模型的维内加权（只对存在的成员归一化权重）
    scores = {}
    for m in all_metrics:
        present = [(ftype, key, direction, mw) for ftype, key, direction, mw in members
                   if m in (per_member.get(key) or {})]
        if not present:
            continue
        wsum = sum(mw for (_ft, _k, _d, mw) in present)
        s = 0.0
        for ftype, key, direction, mw in present:
            s += (mw / wsum) * per_member[key][m]
        scores[m] = s
    return scores


def total_scores(all_metrics, gating=True):
    """六维加权 + 门控 + 缺失容错。返回 (scores, gate_map, dims_map, present_dims)。

    - 整维缺失 → 该维权重在所有存在维上重归一化，总分仍落 0-100
    - 门控维度缺失时（连 D2 都没有）→ 门控自动禁用（R=1）
    """
    dim_map = {}  # dim_name -> {model: dim_score}
    for dim in SCHEME["dims"]:
        name, w, _gated, members = dim
        s = dim_scores(all_metrics, dim)
        if s:
            dim_map[name] = s

    present_dims = list(dim_map.keys())
    wsum = sum(DIM_WEIGHTS[d] for d in present_dims)
    if wsum == 0:
        return {}, {}, {}, []

    R = None
    if gating:
        gate_dim = SCHEME["gate"]
        if gate_dim in dim_map:
            # 门控 R = D2 得分（clamp 0-1）
            R = {m: max(0.0, min(1.0, v)) for m, v in dim_map[gate_dim].items()}

    scores = {}
    for m in all_metrics:
        total = 0.0
        for d in present_dims:
            v = dim_map[d].get(m)
            if v is None:
                continue
            w_n = DIM_WEIGHTS[d] / wsum
            if R is not None and d in GATED_DIMS:
                v = v * R.get(m, 0.0)
            total += w_n * v
        scores[m] = total
    return scores, R, dim_map, present_dims


# ============================================================
# 校准（Spearman ρ + 方向一致率）
# ============================================================
def calibrate(all_metrics, human):
    """对每个候选指标算 Spearman ρ（vs 人工分）与"组内最优是否==人工最优"。

    返回每项:
      ρ, p, 方向一致?, 组内最优模型, 最优==人工最优?, 有偏判据
    close 类指标用"到目标的距离"求 ρ（期望负相关）。
    """
    human_best = max(human, key=human.get)
    rows = []
    for ftype, key, label, direction, in_scheme, belong in CANDIDATE_METRICS:
        raw = {m: v for m, v in ((m, all_metrics[m].get(key)) for m in all_metrics)
               if v is not None and m in human}
        if len(raw) < 3:
            rows.append({"key": key, "label": label, "direction": direction,
                         "in_scheme": in_scheme, "belong": belong, "n": len(raw),
                         "rho": None, "p": None, "consistent": None,
                         "best_model": None, "best_is_human": None})
            continue
        models = list(raw.keys())
        if direction == "close1":
            x = [abs(raw[m] - 1.0) for m in models]
            expected_neg = True
        elif direction == "close05":
            x = [abs(raw[m] - 0.5) for m in models]
            expected_neg = True
        else:
            x = [raw[m] for m in models]
            expected_neg = (direction == "lower")
        y = [human[m] for m in models]
        rho, p = st.spearmanr(x, y)
        if np.isnan(rho):
            rho, p = None, None
        consistent = None if rho is None else (rho < 0 if expected_neg else rho > 0)
        # 组内最优
        if direction == "lower":
            best = min(models, key=lambda m: raw[m])
        elif direction == "higher":
            best = max(models, key=lambda m: raw[m])
        elif direction == "close1":
            best = min(models, key=lambda m: abs(raw[m] - 1.0))
        else:
            best = min(models, key=lambda m: abs(raw[m] - 0.5))
        rows.append({"key": key, "label": label, "direction": direction,
                     "in_scheme": in_scheme, "belong": belong, "n": len(raw),
                     "rho": rho, "p": p, "consistent": consistent,
                     "best_model": best, "best_is_human": (best == human_best)})
    return rows


# ============================================================
# 输出
# ============================================================
def write_scores(out_path, payload):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写入 {out_path}")


def print_calibration(rows):
    print("\n== 指标-人工分校准表（Spearman ρ vs 人工分，N=6）==")
    hdr = f"{'指标':<18}{'方向':<7}{'ρ':>7}{'一致':>5}{'组内最优':<14}{'=人工最优':>8}  判据"
    print(hdr)
    for r in rows:
        if r["rho"] is None:
            continue
        rho_s = f"{r['rho']:+.3f}" + ("*" if r["p"] is not None and r["p"] < 0.1 else "")
        consist = "✓" if r["consistent"] else ("✗" if r["consistent"] is not None else "-")
        best_is = "✓" if r["best_is_human"] else "✗"
        verdict = "计分" if r["in_scheme"] else "参考"
        verdict_s = f"{verdict}({r['belong']})"
        print(f"{r['label']:<18}{r['direction']:<7}{rho_s:>7}{consist:>5}"
              f"{MODEL_LABELS.get(r['best_model'] or '', r['best_model'] or ''):<14}{best_is:>8}  {verdict_s}")


def print_scores(scores, dim_map, human):
    order = sorted(scores, key=scores.get, reverse=True)
    print("\n== 综合评分（0-100，含门控）==")
    print(f"{'模型':<16}{'总分':>8}{'人工分':>8}  " + "  ".join(d for d, *_ in SCHEME["dims"]))
    for m in order:
        dims_s = "  ".join(f"{dim_map[d].get(m, float('nan')):.2f}" for d, *_ in SCHEME["dims"] if d in dim_map)
        print(f"{MODEL_LABELS.get(m, m):<16}{scores[m] * 100:>8.1f}"
              f"{human.get(m, float('nan')):>8.1f}  {dims_s}")
    # 与人工排序的 Kendall τ（如实报告）
    hm = {m: v for m, v in human.items() if m in scores}
    if len(hm) >= 3:
        auto = [scores[m] for m in hm]
        hmv = list(hm.values())
        tau, _ = st.kendalltau(auto, hmv)
        print(f"\n自动总分 vs 人工分 Kendall τ = {tau:.3f}（N={len(hm)} 小样本，含并列时取 τ_b 校正；τ 分辨力有限，参考为主）")


def draw_scorecard(scores, dim_map, human, outpath, dpi=150):
    """总分条形（0-100 色带）+ 六维雷达。"""
    order = sorted(scores, key=scores.get, reverse=True)
    vals = [scores[m] * 100 for m in order]
    labels = [MODEL_LABELS.get(m, m) for m in order]

    fig = plt.figure(figsize=(15, 6.2), dpi=dpi)

    # ---- 左：总分条形 + 色带 + 人工分对照 ----
    ax = fig.add_subplot(1, 2, 1)
    # 0-100 质量色带背景
    bands = [(0, 30, "#e74c3c", "失败带"), (30, 50, "#f39c12", "较差"),
             (50, 70, "#f1c40f", "一般"), (70, 85, "#27ae60", "良好"), (85, 100, "#16a085", "优秀")]
    for lo, hi, c, name in bands:
        ax.axhspan(lo, hi, color=c, alpha=0.08)
        if lo == 0:
            ax.text(len(order) - 0.35, hi + 0.4, name, fontsize=7, color="#666",
                    ha="right", va="bottom")
    xs = np.arange(len(order))
    colors = [MODEL_COLORS.get(m, "#95a5a6") for m in order]
    ax.bar(xs, vals, color=colors, width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=9, zorder=4)
    # 人工分对照（虚线散点）
    hv = [human.get(m) for m in order]
    ax.plot(xs, hv, "k--o", markersize=4, alpha=0.5, label="人工分", zorder=5)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("综合评分（0-100）")
    ax.set_title("综合评分 vs 人工分（门控后）", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- 右：六维雷达 ----
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    dims = [d for d, *_ in SCHEME["dims"] if d in dim_map]
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    for m in order:
        vs = [dim_map[d].get(m, float("nan")) for d in dims]
        if any(np.isnan(v) for v in vs):
            continue
        vs += vs[:1]
        ax2.plot(angles, vs, linewidth=1.6, label=MODEL_LABELS.get(m, m),
                 color=MODEL_COLORS.get(m, "#95a5a6"))
        ax2.fill(angles, vs, alpha=0.05, color=MODEL_COLORS.get(m, "#95a5a6"))
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(dims, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=7, color="#999")
    ax2.set_title("六维归一化画像（1 = 组内最佳）", fontsize=12, pad=18)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [scorecard] 写入 {outpath}")


def main():
    ap = argparse.ArgumentParser(description="综合评分方案（人工分校准 + 0-100 总分）")
    ap.add_argument("--datadir", default="eval_data", help="评估 JSON 目录（默认 eval_data/）")
    ap.add_argument("--models", default=None,
                    help="只看指定模型子集（逗号分隔，默认全部）")
    ap.add_argument("--no-gate", action="store_true", help="对照输出未门控的原始分")
    ap.add_argument("--scorecard", action="store_true", help="画 scorecard.png")
    ap.add_argument("--outdir", default="research-report/figures", help="scorecard 输出目录")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    discovered = discover_models(args.datadir)
    if not discovered:
        print(f"[错误] {args.datadir} 下没有找到 *_metrics.json")
        sys.exit(1)
    order = model_order(discovered)
    if args.models:
        subset = [m for m in args.models.split(",") if m]
        order = [m for m in order if m in subset]
        missing = [m for m in subset if m not in discovered]
        if missing:
            print(f"[警告] 指定模型不存在: {missing}")

    all_metrics = {m: merged_metrics(args.datadir, m) for m in order}
    print(f"模型 ({len(order)}): {', '.join(MODEL_LABELS.get(m, m) for m in order)}")

    # 人工分（缺失不影响打分，仅不参与校准）
    human = {m: HUMAN_SCORE[m] for m in order if m in HUMAN_SCORE}

    # ---- 校准 ----
    rows = calibrate(all_metrics, human)
    print_calibration(rows)

    # ---- 六维 + 门控 ----
    scores, R, dim_map, present_dims = total_scores(all_metrics, gating=not args.no_gate)
    if not scores:
        print("[错误] 没有可计算的维度")
        sys.exit(1)
    print_scores(scores, dim_map, human)

    # ---- 与人工排序的 Kendall τ ----
    hm = {m: v for m, v in human.items() if m in scores}
    tau = None
    if len(hm) >= 3:
        tau = st.kendalltau([scores[m] for m in hm], list(hm.values()))[0]

    # ---- _scores.json ----
    payload = {
        "formula_version": FORMULA_VERSION,
        "gate": FORMULA_GATE,
        "n_models": len(order),
        "weights": DIM_WEIGHTS,
        "gated_dims": sorted(GATED_DIMS),
        "dimensions": [
            {"name": name, "weight": w, "gated": gated,
             "metrics": [{"src": ft, "key": k, "direction": d, "weight": mw}
                         for ft, k, d, mw in members]}
            for name, w, gated, members in SCHEME["dims"]
        ],
        "scores": {m: round(scores[m] * 100, 2) for m in order},
        "dims_scores": {m: {d: _r4(dim_map[d].get(m)) for d in dim_map}
                        for m in order},
        "gate_value": None if R is None else {m: round(v, 4) for m, v in R.items()},
        "tau_vs_human": None if tau is None else round(float(tau), 4),
        "human": {m: v for m, v in human.items()},
        "no_gate": None,
    }
    if args.no_gate:
        ng_scores, *_ = total_scores(all_metrics, gating=False)
        payload["no_gate"] = {m: round(ng_scores[m] * 100, 2) for m in order}
    write_scores(os.path.join(args.datadir, "_scores.json"), payload)

    # ---- scorecard 图 ----
    if args.scorecard:
        os.makedirs(args.outdir, exist_ok=True)
        draw_scorecard(scores, dim_map, human,
                       os.path.join(args.outdir, "scorecard.png"), args.dpi)


if __name__ == "__main__":
    main()
