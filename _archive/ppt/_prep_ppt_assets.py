# -*- coding: utf-8 -*-
"""暑期总结 PPT —— 素材预处理。

把所有要放进 PPT 的拼图一次性生成到 research-report/ppt_assets/。
本脚本只读原始素材、只写 ppt_assets/，可反复重跑。

用法：
    python research-report/_prep_ppt_assets.py
"""
from __future__ import annotations

import glob
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research-report", "ppt_assets")

FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
FONT_REG = "C:/Windows/Fonts/msyh.ttc"

# 配色，与 _build_summer_ppt.py 保持一致
INK = (31, 41, 51)
PRIMARY = (18, 58, 94)
ACCENT = (228, 87, 46)
MUTED = (90, 100, 114)

DD = "generate_project/deep_learning/Fundus-Diffusion/ddpm"


def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def lab(t) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD, t)


def load(path: str, size: int | tuple[int, int] | None = None) -> Image.Image:
    """读图并中心裁成正方形，再缩放到 size（整数=边长，元组=显式宽高）。"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if size is not None:
        target = (size, size) if isinstance(size, int) else size
        im = im.resize(target, Image.LANCZOS)
    return im


def sheet(
    tiles: list[Image.Image],
    cols: int,
    cell: int,
    gap: int = 8,
    bg=(255, 255, 255),
) -> Image.Image:
    """把一批同尺寸方块拼成网格。"""
    rows = (len(tiles) + cols - 1) // cols
    W = cols * cell + (cols + 1) * gap
    H = rows * cell + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (cell + gap)
        y = gap + r * (cell + gap)
        canvas.paste(t.resize((cell, cell), Image.LANCZOS), (x, y))
    return canvas


def strip(
    pairs: list[tuple[Image.Image, str]],
    cell: int,
    gap: int = 16,
    label_h: int = 44,
    bg=(255, 255, 255),
    label_color=MUTED,
    foot: str | None = None,
    foot_color=MUTED,
) -> Image.Image:
    """横排若干个 (图, 下方标签) 单元。标签过宽时自动缩字号，绝不越格。"""
    foot_h = 36 if foot else 0
    W = len(pairs) * cell + (len(pairs) + 1) * gap
    H = cell + label_h + 2 * gap + foot_h
    canvas = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(canvas)
    for i, (im, text) in enumerate(pairs):
        x = gap + i * (cell + gap)
        canvas.paste(im.resize((cell, cell), Image.LANCZOS), (x, gap))
        if not text:
            continue
        size = 26
        while size > 14:
            f = lab(size)
            bb = d.textbbox((0, 0), text, font=f)
            if bb[2] - bb[0] <= cell:
                break
            size -= 1
        f = lab(size)
        bb = d.textbbox((0, 0), text, font=f)
        d.text((x + (cell - (bb[2] - bb[0])) // 2, gap + cell + 8), text, font=f, fill=label_color)
    if foot:
        d.text((gap, H - foot_h + 6), foot, font=lab(22), fill=foot_color)
    return canvas


# --------------------------------------------------------------------------
# 各页素材
# --------------------------------------------------------------------------

def a01_cover() -> None:
    """封面：右侧真实眼底图，左侧深色渐变压暗。"""
    W, H = 2560, 1440
    src = p("fundus/_all_images_ORIGINAL/HRSRC_CASE_102.jpg")
    photo = load(src, size=(H, H)).resize((int(H * 1.15), H), Image.LANCZOS)

    canvas = Image.new("RGB", (W, H), PRIMARY)
    # 把照片贴到右侧，超出部分自然裁掉
    canvas.paste(photo, (W - photo.width + 120, 0))

    # 左侧渐变压暗，让标题有足够对比度
    grad = Image.new("L", (1400, 1))
    for x in range(1400):
        grad.putpixel((x, 0), int(255 * max(0.0, 1.0 - x / 1150.0) ** 0.75))
    grad = grad.resize((1400, H))
    dark = Image.new("RGB", (1400, H), PRIMARY)
    canvas.paste(Image.composite(dark, canvas.crop((0, 0, 1400, H)), grad), (0, 0))

    canvas.save(p("research-report/ppt_assets/p01_cover.png"))
    print("  p01_cover.png", canvas.size, "←", os.path.basename(src))


def a02_real3() -> None:
    """问题页：3 张高分辨率重度病变眼底图。"""
    d = p("fundus/HRSG_4_758_v0.3.4/train/severe")
    # 数据集自带一次几何增广（*_aug_1 后缀），要排掉，否则会挑到同一张图的两个副本
    files = [
        f
        for f in sorted(glob.glob(os.path.join(d, "*.jpg")))
        if "_aug_" not in os.path.basename(f)
    ]
    if len(files) < 3:
        files = sorted(glob.glob(os.path.join(d, "*.png")))[:3]
    files = files[:3]
    tiles = [load(f, 360) for f in files]
    im = sheet(tiles, cols=3, cell=360, gap=14)
    im.save(p("research-report/ppt_assets/p02_real3.png"))
    print("  p02_real3.png", im.size, "←", [os.path.basename(f) for f in files])


def a03_pair() -> None:
    """任务定义：真实图 + 它自己的血管地图（文件名校验过，是真实配对）。"""
    name = "HRSRC_ATLAS_151_A"
    real = load(p(f"fundus/_all_images_ORIGINAL/{name}.jpg"), 340)
    mask = load(p(f"{DD}/conditions/{name}.png"), 340)
    im = strip([(real, "真实眼底照片"), (mask, "从它算出的血管地图")], cell=340)
    im.save(p("research-report/ppt_assets/p03_pair.png"))
    print("  p03_pair.png", im.size, f"← {name} (真实配对)")


def a04_vae_average() -> None:
    """VAE 为什么糊：三张真图 + 它们的逐像素平均（不需要模型就能复现）。"""
    srcs = [p(f"eval_data/real/real_000{i}.png") for i in (0, 1, 2)]
    ims = [load(s, 300) for s in srcs]
    avg = np.mean([np.asarray(i, dtype=np.float32) for i in ims], axis=0).astype(np.uint8)
    avg_im = Image.fromarray(avg)
    im = strip(
        [(ims[0], ""), (ims[1], ""), (ims[2], ""), (avg_im, "三张平均 → 糊了")],
        cell=300,
        label_h=48,
    )
    im.save(p("research-report/ppt_assets/p04_vae_average.png"))
    print("  p04_vae_average.png", im.size, "← eval_data/real/real_0000-0002 像素平均")


def a05_gan() -> None:
    """GAN 全崩：DCGAN 三个训练阶段 + WGAN-GP 末期。"""
    picks = [
        (p("generate_project/deep_learning/Fundus-GAN/dcgan/results_220726_020708/images/epoch_0500.png"), "DCGAN 第 500 轮"),
        (p("generate_project/deep_learning/Fundus-GAN/dcgan/results_220726_020708/images/epoch_1500.png"), "DCGAN 第 1500 轮"),
        (p("generate_project/deep_learning/Fundus-GAN/wgan_gp/results_220726_033306/images/epoch_2750.png"), "WGAN-GP 第 2750 轮"),
    ]
    pairs = []
    for f, t in picks:
        if os.path.exists(f):
            pairs.append((load(f, 330), t))
    im = strip(pairs, cell=330)
    im.save(p("research-report/ppt_assets/p05_gan.png"))
    print("  p05_gan.png", im.size, f"← {len(pairs)} 张")


def a06_bug_pair() -> None:
    """踩坑：颜色映射 bug 修复前 / 后。"""
    d = p("generate_project/deep_learning/Fundus-GAN/dcgan/diagnostic")
    pairs = [
        (load(p(f"{d}/old_normalize_true.png"), 330), "改代码前：暗红"),
        (load(p(f"{d}/new_manual_mapping.png"), 330), "改代码后：变灰了"),
    ]
    im = strip(pairs, cell=330)
    im.save(p("research-report/ppt_assets/p06_bug_pair.png"))
    print("  p06_bug_pair.png", im.size, "← dcgan/diagnostic/")


def a08_noise() -> None:
    """扩散在做什么：把真图逐级加噪、再倒过来擦干净。"""
    base = load(p("eval_data/real/real_0001.png"), 256).resize((300, 300), Image.LANCZOS)
    arr = np.asarray(base, dtype=np.float32)
    rng = np.random.default_rng(0)
    levels = [0.0, 0.15, 0.35, 0.6, 0.85, 1.0]
    ims = []
    for lv in levels:
        noise = rng.normal(0, 255 * 0.42, arr.shape)
        out = np.clip(arr * (1 - lv) + noise * lv, 0, 255).astype(np.uint8)
        ims.append(Image.fromarray(out))
    ims = ims[::-1]  # 纯噪声在左，原图在右
    labels = ["纯噪声", "", "", "", "", "真实图"]
    pairs = list(zip(ims, labels))
    # 说明文字放底部 foot 区，不要压在图片上
    im = strip(pairs, cell=300, gap=10, label_h=48,
               foot="模型学的方向：从纯噪声一路擦成真实图 →",
               foot_color=ACCENT)
    im.save(p("research-report/ppt_assets/p08_noise.png"))
    print("  p08_noise.png", im.size, "← real_0001 逐级加噪")


def a09_dissolve() -> None:
    """病灶溶解：真实（有病灶）vs 早期版本生成（病灶不见了）。

    注意：三张不是同一实例，页面上必须标注"示意"。
    """
    real_f = p("eval_data/real/real_0001.png")
    mse_f = p("eval_data/film/singles/sample_0004.png")
    good_f = p("eval_data/film_l1lpips/singles/sample_0027.png")
    pairs = [
        (load(real_f, 320), "真实照片（有病灶）"),
        (load(mse_f, 320), "早期版本生成（病灶没了）"),
        (load(good_f, 320), "最终版本生成（病灶保住）"),
    ]
    im = strip(pairs, cell=320, foot="三张不是同一个病例，仅作示意")
    im.save(p("research-report/ppt_assets/p09_dissolve.png"))
    print("  p09_dissolve.png", im.size, "← real_0001 / film sample_0004 / l1lpips sample_0035")


def a10_mask_gen_real() -> None:
    """条件生成：血管地图 → 生成 → 真实。同样标注示意。"""
    name = "HRSRC_ATLAS_151_A"
    pairs = [
        (load(p(f"{DD}/conditions/{name}.png"), 320), "① 血管分布图（很粗糙）"),
        (load(p("eval_data/film_l1lpips/singles/sample_0021.png"), 320), "② 模型生成"),
        (load(p(f"fundus/_all_images_ORIGINAL/{name}.jpg"), 320), "③ 真实照片"),
    ]
    im = strip(pairs, cell=320, foot="② 是模型另画的样本，不是 ③ 的复制，仅作示意")
    im.save(p("research-report/ppt_assets/p10_mask_gen_real.png"))
    print("  p10_mask_gen_real.png", im.size, "← mask 151 / l1lpips sample_0005 / 原图 151")


def a11_best() -> None:
    """最佳结果：人工挑过的 8 张干净样本（2×4）。

    刻意不用 result_film_l1lpips.png —— 那张图里含纯噪点样本，
    而且图上印着六维总分 45.2，会和口播的"人工 85-90 分"打架。
    """
    picks = [0, 2, 5, 12, 16, 19, 21, 27]
    tiles = []
    for n in picks:
        f = p(f"eval_data/film_l1lpips/singles/sample_{n:04d}.png")
        if os.path.exists(f):
            tiles.append(load(f, 300))
    im = sheet(tiles, cols=4, cell=300, gap=12)
    out = Image.new("RGB", (im.width, im.height + 36), (255, 255, 255))
    out.paste(im, (0, 0))
    ImageDraw.Draw(out).text((16, im.height + 4), "从 300 张里挑出的干净样本", font=lab(20), fill=(150, 150, 150))
    out.save(p("research-report/ppt_assets/p11_best.png"))
    print("  p11_best.png", im.size, f"← film_l1lpips 手工挑选 {len(tiles)} 张 {picks}")


def a14_comparison_crop() -> None:
    """把竖版王牌图裁掉左侧中文标签列和顶部标题，只留图区。

    原图 908×1410；行边界与列边界是实测量出来的。
    标签改由 pptx 用大字重写，避免缩到 7pt 看不清。
    """
    src = p("generate_project/machine_learning/report/figures/trad-ml-vs-deep_comparison.png")
    im = Image.open(src).convert("RGB")
    # 图区：x 332..896，y 66..1398
    grid = im.crop((332, 66, 896, 1398))
    grid.save(p("research-report/ppt_assets/p13_comparison_grid.png"))
    print("  p13_comparison_grid.png", grid.size, "← 裁自 trad-ml-vs-deep_comparison.png")


def main() -> int:
    if not os.path.exists(FONT_BOLD):
        print(f"错误：找不到中文字体 {FONT_BOLD}", file=sys.stderr)
        return 1
    os.makedirs(OUT, exist_ok=True)
    print(f"输出目录：{os.path.relpath(OUT, ROOT)}")
    for fn in (
        a01_cover, a02_real3, a03_pair, a04_vae_average, a05_gan, a06_bug_pair,
        a08_noise, a09_dissolve, a10_mask_gen_real, a11_best, a14_comparison_crop,
    ):
        fn()
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
