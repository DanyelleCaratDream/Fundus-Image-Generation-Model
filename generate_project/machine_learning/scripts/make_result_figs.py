# -*- coding: utf-8 -*-
"""
make_result_figs.py —— 每方法 3-6 张实验结果合并图（供 REPORT_ML 各方法章嵌入）

用法（在 generate_project/machine_learning/ 下执行）：
    python scripts/make_result_figs.py

输出：report/figures/result_<method>.png
    pca / gmm / patch / film_l1lpips：2×3 网格，6 张 sample 单图（各标 sample 号）
    poisson / retinex：4 组 donor｜底图｜生成 对比条（更能看出"相似但不相同"）
分数为六维门控综合分（11 模型现版 _scores.json，min-max 相对分）。
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
EVAL = os.path.join(ROOT, "eval_data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures")

TITLES = {
    "pca": "PCA 线性重建（基线 1，11.9 分）· 6 张生成样本",
    "gmm": "GMM 混合采样（基线 2，22.6 分）· 6 张生成样本",
    "patch": "补丁拼接（基线 3，31.0 分）· 6 张生成样本",
    "film_l1lpips": "FiLM 扩散（深度学习最佳，45.2 分）· 6 张生成样本（对照）",
    "poisson": "泊松病变重排（EX-004，80.8 分）· donor｜底图｜生成 对比",
    "retinex": "Retinex 光照交换（EX-005，72.5 分）· donor｜底图｜生成 对比",
}
SAMPLE_IDX = [0, 3, 7, 12, 18, 25]      # 6 张，隔开显多样性
MONTAGE = {
    "poisson": ("_montage_v6",
                ["m000_base102_donor270.png", "m002_base219_donor053.png",
                 "m003_base134_donor194.png", "m005_base214_donor050.png"]),
    "retinex": ("_montage",
                ["m000_base102_donor270_a0.82.png", "m001_base188_donor020_a0.65.png",
                 "m003_base151_donor130_a0.61.png", "m004_base257_donor293_a0.60.png"]),
}


def load_font(size):
    for name in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def grid_singles(method):
    """2×3 网格，6 张 sample 单图。"""
    CELL, PAD = 180, 14
    TITLE_H, LABEL_H = 52, 24
    W = 3 * CELL + 4 * PAD
    H = TITLE_H + 2 * (CELL + LABEL_H + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 14), TITLES[method], font=load_font(18), fill=(30, 30, 30))
    folder = os.path.join(EVAL, method, "singles")
    missing = []
    for k, idx in enumerate(SAMPLE_IDX):
        r, c = divmod(k, 3)
        x = PAD + c * (CELL + PAD)
        y = TITLE_H + PAD + r * (CELL + LABEL_H + PAD)
        p = os.path.join(folder, f"sample_{idx:04d}.png")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(im, (x, y))
        else:
            missing.append(p)
        draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(190, 190, 190))
        draw.text((x, y + CELL + 3), f"sample_{idx:04d}", font=load_font(13), fill=(90, 90, 90))
    return canvas, missing


def grid_montage(method):
    """4 行 donor｜底图｜生成 对比条。"""
    STRIP_W = 620                      # 对比条统一缩放宽
    PAD = 14
    TITLE_H = 52
    LABEL_H = 22
    folder, files = MONTAGE[method]
    strip_h = int(STRIP_W * 192 / 576)  # 576×192 等比
    W = STRIP_W + 2 * PAD
    H = TITLE_H + len(files) * (strip_h + LABEL_H + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 14), TITLES[method], font=load_font(18), fill=(30, 30, 30))
    src = os.path.join(EVAL, method, folder)
    missing = []
    for k, f in enumerate(files):
        y = TITLE_H + PAD + k * (strip_h + LABEL_H + PAD)
        p = os.path.join(src, f)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((STRIP_W, strip_h), Image.LANCZOS)
            canvas.paste(im, (PAD, y))
        else:
            missing.append(p)
        draw.rectangle([PAD, y, PAD + STRIP_W - 1, y + strip_h - 1], outline=(190, 190, 190))
        draw.text((PAD, y + strip_h + 3), f, font=load_font(13), fill=(90, 90, 90))
    return canvas, missing


def main():
    os.makedirs(OUT, exist_ok=True)
    for method in ["pca", "gmm", "patch", "film_l1lpips", "poisson", "retinex"]:
        if method in MONTAGE:
            canvas, missing = grid_montage(method)
        else:
            canvas, missing = grid_singles(method)
        out = os.path.join(OUT, f"result_{method}.png")
        canvas.save(out)
        status = "缺 " + ",".join(os.path.basename(m) for m in missing) if missing else "OK"
        print(f"result_{method}.png（{canvas.size[0]}×{canvas.size[1]}）{status}")


if __name__ == "__main__":
    main()
