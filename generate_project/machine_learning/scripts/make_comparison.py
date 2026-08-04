# -*- coding: utf-8 -*-
"""
make_comparison.py —— 报告用五方法并排对比图（面向小白）

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/make_comparison.py

输出：report/figures/trad-ml-vs-deep_comparison.png
    行 = 方法（真实 / 深度最佳 / 3 个传统 ML），列 = 3 个样本索引。

说明：PCA / GMM / 补丁是 Phase C2 的传统 ML 结果（seed=42，各 300 张），
      FiLM+LPIPS 是 Phase A/B 的深度最佳（72.9 分）。分数为六维门控综合分。
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
EVAL = os.path.join(ROOT, "eval_data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures")

ROWS = [
    ("真实原图（330 张）", os.path.join(EVAL, "real"), "real_%04d.png"),
    ("FiLM 扩散（深度学习，72.9 分）", os.path.join(EVAL, "film_l1lpips", "singles"), "sample_%04d.png"),
    ("PCA（传统 ML，6.7 分）", os.path.join(EVAL, "pca", "singles"), "sample_%04d.png"),
    ("GMM（传统 ML，30.4 分）", os.path.join(EVAL, "gmm", "singles"), "sample_%04d.png"),
    ("补丁拼接（传统 ML·纯像素重排，62.9 分）", os.path.join(EVAL, "patch", "singles"), "sample_%04d.png"),
]
IDX = [0, 1, 2]          # 3 个样本
CELL = 180               # 每张缩略图边长
LABEL_W = 300            # 左侧标签区宽度
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
    draw.text((PAD, PAD), "传统 ML vs 深度学习：眼底重度 DR 图生成效果对比（128×128）",
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
