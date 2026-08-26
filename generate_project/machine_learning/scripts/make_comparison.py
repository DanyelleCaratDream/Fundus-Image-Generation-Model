# -*- coding: utf-8 -*-
"""
make_comparison.py —— 报告用七方法并排对比图（面向小白）

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/make_comparison.py

输出：report/figures/trad-ml-vs-deep_comparison.png
    行 = 方法（真实 / 深度最佳 / 2 组合 / 3 传统 ML 基线），列 = 3 个样本索引。

说明：分数为六维门控综合分（11 模型现版 _scores.json，min-max 相对分）。
      poisson/retinex 为 Phase C2 阶段 2 组合方法，pca/gmm/patch 为阶段 1 基线。
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
EVAL = os.path.join(ROOT, "eval_data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures")

ROWS = [
    ("真实原图（330 张）", os.path.join(EVAL, "real"), "real_%04d.png"),
    ("FiLM 扩散（深度学习，45.2 分）", os.path.join(EVAL, "film_l1lpips", "singles"), "sample_%04d.png"),
    ("泊松病变重排（组合·80.8 分）", os.path.join(EVAL, "poisson", "singles"), "sample_%04d.png"),
    ("Retinex 光照交换（组合·72.5 分）", os.path.join(EVAL, "retinex", "singles"), "sample_%04d.png"),
    ("补丁拼接（基线·31.0 分）", os.path.join(EVAL, "patch", "singles"), "sample_%04d.png"),
    ("GMM（基线·22.6 分）", os.path.join(EVAL, "gmm", "singles"), "sample_%04d.png"),
    ("PCA（基线·11.9 分）", os.path.join(EVAL, "pca", "singles"), "sample_%04d.png"),
]
IDX = [0, 1, 2]          # 3 个样本
CELL = 180               # 每张缩略图边长
LABEL_W = 320            # 左侧标签区宽度
PAD = 12                 # 单元间距
TITLE_H = 54             # 顶部标题区


def load_font(size):
    for name in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    os.makedirs(OUT, exist_ok=True)
    f_title = load_font(20)
    f_label = load_font(17)

    n_rows, n_cols = len(ROWS), len(IDX)
    W = LABEL_W + n_cols * CELL + (n_cols + 1) * PAD
    H = TITLE_H + n_rows * CELL + (n_rows + 1) * PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, PAD), "眼底重度 DR 图生成：深度学习 vs 传统 ML 组合 vs 基线（128×128，现版综合分）",
              font=f_title, fill=(20, 20, 20))

    missing = []
    for r, (label, folder, fmt) in enumerate(ROWS):
        y = TITLE_H + PAD + r * (CELL + PAD)
        draw.text((PAD, y + CELL // 2 - 10), label, font=f_label, fill=(40, 40, 40))
        for c, i in enumerate(IDX):
            x = LABEL_W + PAD + c * (CELL + PAD)
            p = os.path.join(folder, fmt % i)
            if not os.path.exists(p):
                missing.append(p)
                continue
            im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(im, (x, y))
            draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(180, 180, 180))

    if missing:
        print("缺失文件（已跳过）:")
        for p in missing:
            print("  " + p)

    out = os.path.join(OUT, "trad-ml-vs-deep_comparison.png")
    canvas.save(out)
    print(f"对比图已保存: {out}（{W}×{H}）")


if __name__ == "__main__":
    main()
