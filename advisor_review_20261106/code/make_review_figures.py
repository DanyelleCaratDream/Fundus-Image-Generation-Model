# -*- coding: utf-8 -*-
"""Rebuild the two advisor contact sheets with Chinese-capable fonts.

Outputs (overwrite):
    advisor_review_20261106/01_compare_sheet.png   (2450 x 1112)
    advisor_review_20261106/02_lesion_strength_sheet.png

Design notes
------------
* Labels are drawn with Microsoft YaHei (msyh.ttc), falling back to SimHei.
  PIL never warns about missing glyphs, so ``load_font`` asserts that the file
  exists, that every string we draw has a non-empty bbox, and that a sample
  glyph has more than one alpha level (a tofu/notdef box is a solid rectangle).
* CFM@128 cells are upsampled 128 -> 256 with BICUBIC for display only; no
  content is altered, the row is labelled 低清对照.
* Figure 01 keeps the original 2450 x 1112 canvas: 48 px title band,
  4 x 256 px rows, 40 px footnote band, 380 px label gutter, 22 px right margin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path("D:/AI_Model_Project_for_Fundus_Color_Images")
OUT_DIR = REPO / "advisor_review_20261106"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

# --------------------------------------------------------------------------
# Strings (kept as module constants so a verifier can grep them)
# --------------------------------------------------------------------------
TITLE_01 = "真实 vs CFM@256 vs 渗出加权 vs CFM@128（同列同序号，非配对）"
LABELS_01 = [
    "真实 HR severe（train，KW-IV）",
    "CFM@256（基线）",
    "CFM@256 + 渗出加权（λ=8；旧结果，受缺陷影响，仅探索）",
    "CFM@128（低清对照）",
]
FOOTNOTE_01 = "各行独立采样；同一列不代表同一患者。"

TITLE_02 = "基于颜色阈值的渗出启发式分数排序（未经医学标注验证）"
LABELS_02 = [
    "最低 8 张（占比均为 0.0，无法区分）",
    "最高 8 张（占比 0.0186–0.0859）",
]
FOOTNOTE_02 = (
    "分数为颜色阈值启发式（min(RGB)>0.42 且 mean>0.48），主要跟随整体亮度；"
    "0 值不代表无病灶，高值不代表真渗出。"
)

# (folder, label, filename stem) - real severe images are named severe_XXXX.png
SHEET_01 = [
    (REPO / "eval_data/real_severe_train", LABELS_01[0], "severe"),
    (REPO / "eval_data/cfm256/singles", LABELS_01[1], "sample"),
    (REPO / "eval_data/cfm256_lesion/singles", LABELS_01[2], "sample"),
    (REPO / "eval_data/cfm128/singles", LABELS_01[3], "sample"),
]

SHEET_02 = [
    (
        REPO / "eval_data/cfm256/singles",
        LABELS_02[0],
        [0, 2, 13, 14, 21, 57, 44, 32],
    ),
    (
        REPO / "eval_data/cfm256/singles",
        LABELS_02[1],
        [56, 55, 23, 54, 51, 58, 36, 41],
    ),
]


# --------------------------------------------------------------------------
# Font handling
# --------------------------------------------------------------------------
def _glyph_ok(font: ImageFont.FreeTypeFont, sizes=(12, 24, 48)) -> bool:
    """True if the font maps CJK text to actually drawn (non-tofu) glyphs."""
    for size in sizes:
        f = font.font_variant(size=size)
        bbox = f.getbbox("渗眼底")
        if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
            return False
        mask = f.getmask("渗")
        data = bytes(mask)
        if not data or max(data) == 0:
            return False
        # A .notdef box is a single flat value; real glyphs have antialiasing.
        if len(set(data)) < 2:
            return False
    return True


def load_font(size: int) -> ImageFont.FreeTypeFont:
    errors = []
    for path in FONT_CANDIDATES:
        if not os.path.isfile(path):
            errors.append(f"{path}: file not found")
            continue
        try:
            font = ImageFont.truetype(path, size)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{path}: {exc}")
            continue
        if _glyph_ok(font):
            font.name_path = path  # type: ignore[attr-defined]
            return font
        errors.append(f"{path}: rendered CJK glyphs look empty/tofu")
    raise SystemExit("No usable Chinese font found:\n  " + "\n  ".join(errors))


def assert_text_renderable(font: ImageFont.FreeTypeFont, strings) -> None:
    """Fail loudly if any string we intend to draw measures to zero width."""
    for s in strings:
        bbox = font.getbbox(s)
        if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
            raise SystemExit(f"String does not measure with {font.name_path!r}: {s!r}")


# --------------------------------------------------------------------------
# Text layout helpers
# --------------------------------------------------------------------------
NO_LINE_START = "，。、；：）】》,.;:)]}%"


def _line_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    return font.getlength(text)


def wrap_text(font: ImageFont.FreeTypeFont, text: str, max_width: int) -> list[str]:
    """Greedy wrap that keeps Latin words intact and avoids leading punctuation."""
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if ch == " ":
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(" ")
        elif ord(ch) > 0x2E80:  # CJK-ish: each char is its own break opportunity
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        else:
            buf += ch
    if buf:
        tokens.append(buf)

    lines: list[str] = []
    cur = ""
    for tok in tokens:
        trial = cur + tok
        if _line_width(font, trial) <= max_width or not cur.strip():
            cur = trial
        else:
            if tok and tok[0] in NO_LINE_START:
                cur = trial  # keep punctuation glued to the previous line
            else:
                lines.append(cur.rstrip())
                cur = "" if tok == " " else tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines or [""]


def line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def draw_block(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    lines: list[str],
    box: tuple[int, int, int, int],
    fill,
    align: str = "left",
) -> None:
    """Draw wrapped lines vertically centred inside box."""
    lh = line_height(font)
    total = lh * len(lines)
    x0, y0, x1, y1 = box
    y = y0 + (y1 - y0 - total) // 2
    for line in lines:
        if align == "center":
            x = x0 + (x1 - x0 - _line_width(font, line)) / 2
        else:
            x = x0
        draw.text((x, y), line, font=font, fill=fill)
        y += lh


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
CELL = 256


def load_cell(path: Path) -> Image.Image:
    if not path.is_file():
        raise SystemExit(f"Missing input image: {path}")
    im = Image.open(path).convert("RGB")
    if im.size != (CELL, CELL):
        im = im.resize((CELL, CELL), Image.Resampling.BICUBIC)  # display only
    return im


def save_and_verify(img: Image.Image, path: Path) -> tuple[int, int]:
    img.save(path, optimize=True)
    with Image.open(path) as chk:
        chk.load()
        size = chk.size
        if size != img.size:
            raise SystemExit(f"{path}: size mismatch after save {size} vs {img.size}")
        if chk.getbbox() is None:
            raise SystemExit(f"{path}: saved image is empty")
        extrema = chk.convert("RGB").getextrema()
        if all(lo == hi for lo, hi in extrema):
            raise SystemExit(f"{path}: saved image is a flat colour")
    if path.stat().st_size < 10_000:
        raise SystemExit(f"{path}: suspiciously small file")
    return size


# --------------------------------------------------------------------------
# Figure 01
# --------------------------------------------------------------------------
def build_figure_01() -> Path:
    w, h = 2450, 1112
    gutter, right_margin = 380, 22
    title_h, foot_h = 48, 40

    f_title = load_font(27)
    f_label = load_font(21)
    f_foot = load_font(18)
    assert_text_renderable(
        f_title, [TITLE_01, TITLE_02, FOOTNOTE_01, FOOTNOTE_02, *LABELS_01, *LABELS_02]
    )
    assert_text_renderable(f_label, [*LABELS_01, *LABELS_02])

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title band
    draw_block(draw, f_title, [TITLE_01], (0, 0, w, title_h), (0, 0, 0), align="center")

    grid_x0 = gutter
    n_rows = len(SHEET_01)

    for r, (folder, label, stem) in enumerate(SHEET_01):
        y0 = title_h + r * CELL
        # Row label gutter
        label_box = (14, y0, gutter - 14, y0 + CELL)
        lines = wrap_text(f_label, label, gutter - 34)
        draw_block(draw, f_label, lines, label_box, (30, 30, 30), align="left")
        # Cells
        for c in range(8):
            src = folder / f"{stem}_{c:04d}.png"
            cell = load_cell(src)
            x = grid_x0 + c * CELL
            img.paste(cell, (x, y0))
            draw.rectangle(
                [x, y0, x + CELL - 1, y0 + CELL - 1], outline=(225, 225, 225), width=1
            )

    # separators
    for r in range(n_rows + 1):
        y = title_h + r * CELL
        draw.line([(0, y), (w, y)], fill=(205, 205, 205), width=1)
    draw.line([(gutter, title_h), (gutter, title_h + n_rows * CELL)], fill=(205, 205, 205))

    # Footnote band
    draw_block(
        draw,
        f_foot,
        [FOOTNOTE_01],
        (14, title_h + n_rows * CELL, w - right_margin, h - 4),
        (90, 90, 90),
        align="left",
    )

    out = OUT_DIR / "01_compare_sheet.png"
    save_and_verify(img, out)
    return out


# --------------------------------------------------------------------------
# Figure 02
# --------------------------------------------------------------------------
def build_figure_02() -> Path:
    w = 2450
    gutter, right_margin = 380, 22
    title_h = 56

    f_title = load_font(27)
    f_label = load_font(21)
    f_foot = load_font(18)

    foot_lines = wrap_text(f_foot, FOOTNOTE_02, w - 40)
    foot_h = line_height(f_foot) * len(foot_lines) + 18
    h = title_h + 2 * CELL + foot_h

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw_block(draw, f_title, [TITLE_02], (0, 0, w, title_h), (0, 0, 0), align="center")

    for r, (folder, label, indices) in enumerate(SHEET_02):
        y0 = title_h + r * CELL
        lines = wrap_text(f_label, label, gutter - 34)
        draw_block(draw, f_label, lines, (14, y0, gutter - 14, y0 + CELL), (30, 30, 30))
        for c, idx in enumerate(indices):
            src = folder / f"sample_{idx:04d}.png"
            img.paste(load_cell(src), (gutter + c * CELL, y0))
            x = gutter + c * CELL
            draw.rectangle(
                [x, y0, x + CELL - 1, y0 + CELL - 1], outline=(225, 225, 225), width=1
            )

    for r in range(3):
        y = title_h + r * CELL
        draw.line([(0, y), (w, y)], fill=(205, 205, 205), width=1)
    draw.line([(gutter, title_h), (gutter, title_h + 2 * CELL)], fill=(205, 205, 205))

    draw_block(
        draw,
        f_foot,
        foot_lines,
        (14, title_h + 2 * CELL, w - right_margin, h - 2),
        (90, 90, 90),
    )

    out = OUT_DIR / "02_lesion_strength_sheet.png"
    save_and_verify(img, out)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (build_figure_01, build_figure_02):
        path = fn()
        with Image.open(path) as im:
            print(f"{path}  {im.size[0]}x{im.size[1]}  {path.stat().st_size} bytes")
    print("font:", load_font(20).name_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
