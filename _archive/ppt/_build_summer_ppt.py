# -*- coding: utf-8 -*-
"""生成暑期总结汇报 PPT（16 页，16:9，含逐字讲稿备注）。

页序：封面 → 问题 → 数据/任务 → VAE → GAN → 两个 bug → 为什么崩 → 扩散 →
      病灶溶解 → 血管条件 → 最终方案 → FiLM/L1/LPIPS 展开 → 评估体系 →
      后续进展(ML) → 总结 → 备用页

依赖：python-pptx、pillow。素材由 _prep_ppt_assets.py 先生成。

用法：
    python research-report/_prep_ppt_assets.py     # 先生成素材
    python research-report/_build_summer_ppt.py    # 再生成 pptx

产出：
    research-report/暑期总结_深度学习.pptx
    research-report/ppt_assets/_preview/pNN.png    # 版面自检预览图
"""
from __future__ import annotations

import io
import os
import sys

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "research-report", "ppt_assets")
OUT_PPTX = os.path.join(ROOT, "research-report", "暑期总结_深度学习.pptx")
PREVIEW_DIR = os.path.join(ASSETS, "_preview")

FONT = "微软雅黑"
FONT_FILE = "C:/Windows/Fonts/msyh.ttc"
# 公式单独用一个字体：微软雅黑没有 x̂ 的字形，会渲染成豆腐块；Cambria 有全套数学符号
MATH = "Cambria"

# ---- 画布与网格（英寸） ----
SW, SH = 13.333, 7.5
M = 0.55                      # 左右边距
TITLE_Y, TITLE_H = 0.30, 0.62
RULE_Y = 0.99
BODY_Y, BODY_H = 1.20, 5.78
BODY_W = SW - 2 * M

# ---- 配色（与图片素材里的绿橙调协调）----
SUB = {"baseline": -25000}     # 下标
SUP = {"baseline": 30000}      # 上标
XHAT = "x" + chr(0x302)   # x̂（x + 组合抑扬符）—— 写成 chr() 而不是字面量，
                       #   免得这个看不见的组合字符被编辑器/工具链吃掉

INK = RGBColor(0x1F, 0x29, 0x33)
PRIMARY = RGBColor(0x12, 0x3A, 0x5E)
ACCENT = RGBColor(0xE4, 0x57, 0x2E)
MUTED = RGBColor(0x5A, 0x64, 0x72)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BAND = RGBColor(0xF2, 0xF5, 0xF8)

A = lambda n: os.path.join(ASSETS, n)


# ==========================================================================
# 底层工具
# ==========================================================================

def set_run_font(run, size=16, bold=False, color=INK, name=FONT, baseline=None, strike=False):
    """中文字体必须同时写 a:latin / a:ea / a:cs —— 只设 font.name 对中文不生效。

    baseline: 30000=上标, -25000=下标（PowerPoint 用千分比表示基线偏移）。
    """
    f = run.font
    f.name = name
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    if baseline is not None:
        rPr.set("baseline", str(baseline))
    if strike:
        rPr.set("strike", "sngStrike")
    latin = rPr.find(qn("a:latin"))
    if latin is None:
        latin = rPr.makeelement(qn("a:latin"), {})
        rPr.append(latin)
    latin.set("typeface", name)
    for tag in ("a:ea", "a:cs"):
        old = rPr.find(qn(tag))
        if old is not None:
            rPr.remove(old)
    cs = latin.makeelement(qn("a:cs"), {"typeface": name})
    ea = latin.makeelement(qn("a:ea"), {"typeface": name})
    latin.addnext(cs)
    latin.addnext(ea)


def add_text(slide, box, paras, anchor=MSO_ANCHOR.TOP, line_spacing=1.30):
    """paras: [{"t": str, ...}] 或 [{"runs": [(text, opts), ...], ...}]

    给了 "runs" 就按 run 逐个排（用于公式的上下标、混排字体、删除线）。
    """
    x, y, w, h = box
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(spec.get("space_after", 6))
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        base = dict(size=spec.get("size", 16), bold=spec.get("bold", False),
                    color=spec.get("color", INK))
        runs = spec.get("runs") or [(spec["t"], {})]
        for text, opts in runs:
            run = p.add_run()
            run.text = text
            set_run_font(run, **{**base, **opts})
    return tb


def fit_contain(path, box):
    """把图等比缩放后居中放进 box，返回 EMU 的 (left, top, width, height)。绝不拉伸。"""
    x, y, w, h = (Inches(v) for v in box)
    with Image.open(path) as im:
        iw, ih = im.size
    ar_img, ar_box = iw / ih, w / h
    if ar_img >= ar_box:
        width, height = w, int(round(w / ar_img))
    else:
        height, width = h, int(round(h * ar_img))
    return int(x + (w - width) // 2), int(y + (h - height) // 2), width, height


def add_pic(slide, path, box):
    l, t, w, h = fit_contain(path, box)
    return slide.shapes.add_picture(path, l, t, width=w, height=h)


def add_rect(slide, box, fill, line=None):
    x, y, w, h = box
    sh = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    sh.adjustments[0] = 0.08
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def set_notes(slide, script):
    tf = slide.notes_slide.notes_text_frame
    tf.text = script
    for p in tf.paragraphs:
        for r in p.runs:
            set_run_font(r, size=12, color=RGBColor(0, 0, 0))


# ==========================================================================
# 页面内容
# ==========================================================================

P1_NOTES = "各位老师好。我这个暑假做的事，一句话说：给我三百多张眼底照片，让电脑学会眼底照片长什么样，再自己画出新的。"

P2_NOTES = "先说为什么做。要让 AI 学会看眼底，需要大量标注图；但越是严重的病变、越罕见的病例，样本越少。所以我们想让模型自己生成新样本补上。"

P3_NOTES = "我们一共 330 张眼底彩照，全部裁到 128 乘 128。我还从每张图里提取了血管分布，就是右边这张黑白图，后面会派上大用场。数据少、图又小，是后面所有困难的源头。"

P4_NOTES = "第一个模型叫 VAE。它的损失函数是两部分相加：一部分逐像素比差距，一部分约束模型别把特征记得太死。问题出在第一部分——逐像素比差距的最优解就是「平均值」，所以它画出来的是张平均脸。想通这个我就知道，加大模型、训练翻倍都没用：损失降了一半，但 800 轮和 1200 轮的图肉眼没区别。"

P5_NOTES = "第二个是 GAN，一个画画、一个挑毛病。但它在这 330 张数据上立刻崩了：DCGAN 全是暗红糊块；WGAN-GP 两轮判别器分数就炸；StyleGAN2 环境跑不通，算下来要跑 275 天。"

P6_NOTES = "崩了不能就算了。我查到两个问题：颜色映射写错，图被渲染成暗红；修完变成灰的——听着像修好了，其实更麻烦。一查像素统计才发现，真正的问题是模型根本没学颜色，绿蓝通道一直卡着不动。"

P7_NOTES = "但把问题都修完，GAN 还是不能用。因为只有 330 张图时，那个挑毛病的会背题——它不是学会判断像不像眼底图，而是记住了这 330 张的细节。瓶颈不是模型不够强，是数据太少。"

P8_NOTES = "于是我换了第三种思路，扩散模型。它的训练目标很不一样：随机给图加噪，然后让模型猜加了多少噪声——擦干净了，图就出来了。这条路不需要裁判，训练稳定得多。但注意，它用的损失函数还是 MSE。"

P9_NOTES = "但马上冒出个新问题。MSE 是对全部像素求误差，而眼底图里 95% 的像素是暗背景，只有 5% 是亮的病灶——把病灶画暗一点，总误差反而更小。所以病灶被「优化」掉了。左边是真实图，中间早期版本生成的，病灶没了。这和第 4 页 VAE 糊掉是同一个原因：MSE 奖励的就是「平均」。"

P10_NOTES = "解决办法是给模型加个约束：把前面算出的血管分布图，训练时作为条件一起喂给它。代价极小，只多了一个卷积层、576 个参数。"

P11_NOTES = "在此基础上我又改了两处：一是网络内部换成一种叫 FiLM 的调制方式，让每一路特征都能单独调节，而不是整体加同一个值；二是把损失函数从 MSE 换成 L1 加感知损失——L1 不奖励平均，感知损失比的是特征而不是逐个像素，血管和病灶的边界就保住了。改完病灶基本保住，早期版本会丢掉约四分之三。"

P_TERMS_NOTES = "这三个词展开说一下。FiLM 是把「所有通道加同一个值」换成「每个通道单独调节」，代价只是参数多 4.6%。L1 和 MSE 的差别在惩罚的形状：平方让大误差的代价翻倍涨，模型就宁可选中庸；L1 按比例罚。LPIPS 不比像素，比的是网络中间层的特征——血管边缘糊一个像素它也重罚。"

P12_NOTES = "做到这步我发现个新问题：我说模型变好了，凭什么？最开始只能自己打 80 分，被问依据就答不上来。所以我搭了一套两层评估：通用层看像不像、有没有多样性；眼底专用层看病灶保没保住、血管对不对。"

P13_NOTES = "最后讲后续。深度模型卡在数据量上之后，我换了个思路：不动模型，直接重新组合真实图里的零件——把一张图的病变区域搬到另一张图上。用同一套评估体系量，它拿到 80.8 分，比深度学习高。但它是在搬运真实像素，不会创造新的病灶形态。"

P14_NOTES = "总结这个暑假。第一，数据只有 330 张时，瓶颈是数据不是模型。第二，失败要查到底——都是能定位的具体原因。第三，评估比生成更难，也更值钱。谢谢老师。"

P15_NOTES = "（备用页，老师追问再翻）如果被问到为什么传统方法分数反而比深度学习高：因为泊松重排和 Retinex 都是在搬运真实图的像素，出来的图当然真、分数当然高；而生成模型是从零合成整张图，它最大的价值恰恰是能造出训练集里没有的病灶形态。两者分数可比，但用途根本不同。下一步打算把生成的图拿去扩充糖尿病视网膜病变分级分类器的 severe 类，验证生成图对下游任务到底有没有用。"


# P13 左侧那七大行对应的中文标签，由 pptx 用大字号重写 ——
# 原图内嵌的标签缩到 16:9 版面里只有约 7pt，投影时后排根本看不清。
_ML_ROWS = [
    ("真实原图", False),
    ("FiLM 扩散（深度学习）", True),
    ("泊松病变重排　80.8", False),
    ("Retinex 光照交换　72.5", False),
    ("补丁拼接　31.0", False),
    ("GMM　22.6", False),
    ("PCA　11.9", False),
]
_GRID_TOP, _GRID_H = 1.30, 5.75      # p13_comparison_grid 的落地位置与高度
_GRID_ROW_PITCH, _GRID_ROW_H = 192, 180   # 原图里每行 180px + 12px 间隔


def p13_row_labels():
    scale = _GRID_H / 1332.0
    out = []
    for i, (text, hi) in enumerate(_ML_ROWS):
        cy = _GRID_TOP + (i * _GRID_ROW_PITCH + _GRID_ROW_H / 2) * scale
        out.append(dict(
            kind="rowlabel",
            box=(7.70, cy - 0.17, 2.32, 0.34),
            t=text,
            color=ACCENT if hi else INK,
            bold=hi,
        ))
    return out


def slides_spec():
    spec = [
        # ---------------- P1 封面 ----------------
        dict(
            n=1, bleed=True, notes=P1_NOTES, title=None,
            items=[
                dict(kind="pic", src=A("p01_cover.png"), box=(0, 0, SW, SH)),
                dict(kind="text", box=(0.95, 2.55, 7.4, 2.6), paras=[
                    dict(t="这个暑假我学了什么", size=40, bold=True, color=WHITE, space_after=10),
                    dict(t="—— 让电脑学会画眼底照片", size=24, color=RGBColor(0xC9, 0xD8, 0xE4), space_after=26),
                    dict(t="本科生暑期学习汇报", size=15, color=RGBColor(0x9F, 0xB4, 0xC6), space_after=4),
                    dict(t="姓名 ｜ 2026 年 9 月", size=15, color=RGBColor(0x9F, 0xB4, 0xC6)),
                ]),
            ],
        ),
        # ---------------- P2 问题 ----------------
        dict(
            n=2, title="问题：眼底照片不够用", notes=P2_NOTES,
            items=[
                dict(kind="pic", src=A("p02_real3.png"), box=(M, 1.28, BODY_W, 3.70)),
                dict(kind="text", box=(M, 5.42, BODY_W, 1.50), paras=[
                    dict(t="要让 AI 学会看眼底、判断有没有病变，需要大量标注好的图。", size=18, space_after=8),
                    dict(t="但越是严重的病变、越罕见的病例，样本越少。", size=18, space_after=8),
                    dict(t="缺的不是算法，是数据。", size=18, bold=True, color=ACCENT, space_after=0),
                ]),
                dict(kind="caption", box=(M, 5.06, BODY_W, 0.28), t="真实眼底照片（重度病变样本）"),
            ],
        ),
        # ---------------- P3 数据与任务 ----------------
        dict(
            n=3, title="我们的数据与任务", notes=P3_NOTES,
            items=[
                dict(kind="pic", src=A("p03_pair.png"), box=(M, 1.75, 7.30, 4.60)),
                dict(kind="text", box=(8.15, 1.75, 4.63, 4.80), paras=[
                    dict(t="330 张眼底彩照", size=19, bold=True, color=PRIMARY, space_after=4),
                    dict(t="全部裁到 128×128 —— 一根血管只剩 1~2 个像素", size=15, color=MUTED, space_after=18),
                    dict(t="要做的事：", size=19, bold=True, color=PRIMARY, space_after=4),
                    dict(t="让模型学会“眼底图长什么样”，然后自己画一张新的", size=16, space_after=18),
                    dict(t="顺手算出每张图的血管分布（右图）", size=15, color=MUTED, space_after=0),
                ]),
            ],
        ),
        # ---------------- P4 VAE ----------------
        dict(
            n=4, title="第一站：VAE —— 学是学会了，但画不清", notes=P4_NOTES,
            items=[
                dict(kind="pic",
                     src=os.path.join(ROOT, "generate_project/deep_learning/Fundus-VAE/"
                                            "results/vanilla_vae/final_generated.png"),
                     box=(M, 1.30, 5.30, 5.45)),
                dict(kind="pic", src=A("p04_vae_average.png"), box=(6.20, 1.28, 6.58, 2.20)),
                dict(kind="formula", box=(6.20, 3.58, 6.58, 0.58), size=22, runs=[
                    ("L = ‖x − " + XHAT + "‖", {}), ("2", SUP), ("  +  β · KL", {}),
                ]),
                dict(kind="text", box=(6.20, 4.20, 6.58, 0.42), paras=[
                    dict(t="逐像素比差距　＋　别把特征记得太死", size=13, color=MUTED,
                         align=PP_ALIGN.CENTER, space_after=0),
                ]),
                dict(kind="text", box=(6.20, 4.78, 6.58, 2.05), paras=[
                    dict(t="MSE 的最优解就是「平均值」——所以它画的是「平均脸」。", size=16,
                         bold=True, color=ACCENT, space_after=10),
                    dict(t="后来模型加大、训练翻倍，损失降了一半 —— 但 800 轮和 1200 轮肉眼没区别。", size=14,
                         color=MUTED, space_after=0),
                ]),
                dict(kind="caption", box=(M, 6.78, 5.30, 0.30), t="64 张 VAE 自己画的"),
            ],
        ),
        # ---------------- P5 GAN ----------------
        dict(
            n=5, title="第二站：GAN —— 彻底崩了", notes=P5_NOTES,
            items=[
                dict(kind="pic", src=A("p05_gan.png"), box=(M, 1.40, BODY_W, 4.05)),
                dict(kind="text", box=(M, 5.60, BODY_W, 1.45), paras=[
                    dict(t="换了三种 GAN，全都没做出来。", size=19, bold=True, color=ACCENT, space_after=8),
                    dict(t="DCGAN 画出来全是暗红糊块；WGAN-GP 训练两轮判别器分数就炸；StyleGAN2 环境跑不通，算下来要 275 天。", size=15, color=MUTED, space_after=0),
                ]),
            ],
        ),
        # ---------------- P6 踩坑 ----------------
        dict(
            n=6, title="崩了就得查：两个具体的问题", notes=P6_NOTES,
            items=[
                dict(kind="pic", src=A("p06_bug_pair.png"), box=(M, 1.55, 6.55, 4.10)),
                dict(kind="text", box=(7.50, 1.55, 5.28, 0.90), paras=[
                    dict(t="① 颜色映射写错 → 图被画成暗红", size=16, space_after=6),
                    dict(t="② 修好后变成灰的 —— 还是不对", size=16, space_after=0),
                ]),
                dict(kind="text", box=(7.50, 2.62, 5.28, 0.55), paras=[
                    dict(t="真正的问题：绿、蓝两个通道根本没学，一直卡在 0.5 不动。", size=15, color=ACCENT, bold=True, space_after=0),
                ]),
                dict(kind="band", box=(7.50, 3.55, 5.28, 0.52), color=RGBColor(0x7B, 0x49, 0x25),
                     label="真实眼底    R 0.48 / G 0.29 / B 0.15", tcolor=WHITE, size=13),
                dict(kind="band", box=(7.50, 4.20, 5.28, 0.52), color=RGBColor(0x68, 0x4B, 0x48),
                     label="改代码前    R 0.41 / G 0.30 / B 0.28", tcolor=WHITE, size=13),
                dict(kind="band", box=(7.50, 4.85, 5.28, 0.52), color=RGBColor(0x96, 0x82, 0x80),
                     label="改代码后    R 0.59 / G 0.51 / B 0.50（灰）", tcolor=WHITE, size=13),
                dict(kind="text", box=(7.50, 5.55, 5.28, 1.10), paras=[
                    dict(t="不是玄学，是两个能定位、能查证的具体问题。", size=15, color=MUTED, space_after=0),
                ]),
                dict(kind="caption", box=(M, 5.75, 6.55, 0.30), t="左：改代码前 ｜ 右：改代码后"),
            ],
        ),
        # ---------------- P7 转折 ----------------
        dict(
            n=7, title="但修好问题，它还是不行 —— 为什么？", notes=P7_NOTES,
            items=[
                dict(kind="rect", box=(1.30, 1.85, 3.10, 1.30), color=RGBColor(0xE8, 0xEF, 0xF6),
                     label="画画的（生成器）", tcolor=PRIMARY, size=19),
                dict(kind="rect", box=(8.90, 1.85, 3.10, 1.30), color=RGBColor(0xFD, 0xED, 0xE8),
                     label="挑毛病的（判别器）", tcolor=ACCENT, size=19),
                dict(kind="text", box=(4.60, 2.02, 4.10, 0.95), paras=[
                    dict(t="画一张  →", size=17, color=MUTED, align=PP_ALIGN.CENTER, space_after=6),
                    dict(t="←  说假", size=17, color=MUTED, align=PP_ALIGN.CENTER, space_after=0),
                ]),
                dict(kind="rect", box=(8.90, 3.50, 3.10, 0.85), color=RGBColor(0xDD, 0xE4, 0xEC),
                     label="把 330 张全背下来了", tcolor=INK, size=16),
                dict(kind="text", box=(1.30, 3.60, 7.10, 0.70), paras=[
                    dict(t="裁判太强 → 画家得不到任何有用的反馈", size=18, bold=True, color=ACCENT, space_after=0),
                ]),
                dict(kind="text", box=(1.30, 4.65, 10.70, 2.05), paras=[
                    dict(t="330 张图太少，判别器不是学会“判断像不像眼底图”，而是记住了这 330 张的细节 —— 你画什么它都认得出来是假的。", size=17, space_after=10),
                    dict(t="生成器因此学不到任何东西。这不是调参能解决的。", size=17, space_after=10),
                    dict(t="瓶颈不是模型不够强，是数据太少。", size=19, bold=True, color=PRIMARY, space_after=0),
                ]),
            ],
        ),
        # ---------------- P8 Diffusion ----------------
        dict(
            n=8, title="第三站：换扩散模型 —— 终于像样了", notes=P8_NOTES,
            items=[
                dict(kind="pic", src=A("p08_noise.png"), box=(M, 1.28, BODY_W, 2.55)),
                dict(kind="formula", box=(M, 3.98, BODY_W, 0.58), size=22, runs=[
                    ("L = ‖ε − ε", {}), ("θ", SUB), ("(x", {}), ("t", SUB),
                    (", t)‖", {}), ("2", SUP),
                ]),
                dict(kind="text", box=(M, 4.60, BODY_W, 0.42), paras=[
                    dict(t="随机加噪，让模型猜「加了多少噪声」　——　注意：这里用的仍然是 MSE", size=13.5,
                         bold=True, color=ACCENT, align=PP_ALIGN.CENTER, space_after=0),
                ]),
                dict(kind="text", box=(M, 5.14, BODY_W, 1.70), paras=[
                    dict(t="换个思路：不学怎么画，先学怎么把噪声擦干净。", size=18, space_after=10),
                    dict(t="不需要“裁判”，训练稳定得多 —— 第一次跑出来血管走向就对得上。", size=16,
                         color=MUTED, space_after=0),
                ]),
            ],
        ),
        # ---------------- P9 病灶溶解 ----------------
        dict(
            n=9, title="新问题：病灶会“溶解”", notes=P9_NOTES,
            items=[
                dict(kind="rect", box=(8.28, 1.28, 4.50, 4.95), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="pic", src=A("p09_dissolve.png"), box=(M, 1.30, 7.45, 3.15)),
                dict(kind="text", box=(M, 4.62, 7.45, 2.25), paras=[
                    dict(t="眼底图里 95% 的像素是暗背景，只有 5% 是亮病灶。", size=16, space_after=10),
                    dict(t="把病灶画暗一点，总误差反而更小 —— 病灶就被「优化」掉了。", size=17,
                         bold=True, color=ACCENT, space_after=10),
                    dict(t="对医学用途，这是致命的。", size=16, color=MUTED, space_after=0),
                ]),
                dict(kind="text", box=(8.50, 1.42, 4.06, 0.38), paras=[
                    dict(t="数学上发生了什么", size=15, bold=True, color=PRIMARY, space_after=0),
                ]),
                dict(kind="formula", box=(8.50, 1.88, 4.06, 0.62), size=20, runs=[
                    ("L = (1/N) · Σ", {}), ("i", SUB), (" (x", {}), ("i", SUB),
                    (" − " + XHAT, {}), ("i", SUB), (")", {}), ("2", SUP),
                ]),
                dict(kind="text", box=(8.50, 2.60, 4.06, 1.70), paras=[
                    dict(t="误差是对「全部像素」求的。", size=14.5, space_after=8),
                    dict(t="暗背景占 95%，票数压倒病灶 —— 把病灶画暗一点，总误差反而更小。", size=14.5,
                         color=MUTED, space_after=0),
                ]),
                dict(kind="text", box=(8.50, 4.35, 4.06, 1.75), paras=[
                    dict(t="所以病灶被一点点抹掉。", size=15.5, bold=True, color=INK, space_after=10),
                    dict(t="这和第 4 页 VAE 糊掉，是同一个原因：", size=15.5, color=MUTED, space_after=4),
                    dict(t="MSE 奖励的就是「平均」。", size=16, bold=True, color=ACCENT, space_after=0),
                ]),
            ],
        ),
        # ---------------- P10 条件生成 ----------------
        dict(
            n=10, title="解决：先给它一张血管地图", notes=P10_NOTES,
            items=[
                dict(kind="pic", src=A("p10_mask_gen_real.png"), box=(M, 1.32, BODY_W, 4.05)),
                dict(kind="text", box=(M, 5.55, BODY_W, 1.30), paras=[
                    dict(t="训练时把血管分布图作为条件一起喂给模型，告诉它血管该长在哪儿。", size=17, space_after=8),
                    dict(t="代价极小：模型只多了一个卷积层、576 个参数。但它现在知道该往哪儿画了。", size=17, color=MUTED, space_after=0),
                ]),
            ],
        ),
        # ---------------- P11 最终方案 ----------------
        dict(
            n=11, title="最终方案与效果", notes=P11_NOTES,
            items=[
                dict(kind="rect", box=(8.02, 1.28, 4.76, 5.28), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="pic", src=A("p11_best.png"), box=(M, 1.55, 7.25, 4.90)),
                dict(kind="text", box=(8.22, 1.42, 4.38, 1.35), paras=[
                    dict(t="改了两处", size=15, bold=True, color=PRIMARY, space_after=8),
                    dict(t="① 网络内部换成 FiLM 调制 —— 每路特征单独调节，不是统一加同一个值",
                         size=14.5, space_after=6),
                    dict(t="② 损失函数不再「求平均」", size=14.5, space_after=0),
                ]),
                dict(kind="formula", box=(8.22, 2.88, 4.38, 0.60), size=20, runs=[
                    ("MSE", {"strike": True, "color": MUTED}),
                    ("  →  ", {"color": MUTED}),
                    ("L1 + LPIPS", {"color": ACCENT, "bold": True, "size": 22}),
                ]),
                dict(kind="text", box=(8.16, 3.54, 4.48, 0.95), paras=[
                    dict(t="L1 不奖励「平均」；感知损失比的是特征，不是逐像素。",
                         size=12.5, color=MUTED, space_after=0),
                ]),
                dict(kind="text", box=(8.22, 4.62, 4.38, 1.90), paras=[
                    dict(t="病灶基本保住（早期版本会丢掉约四分之三）", size=15.5, bold=True, color=ACCENT, space_after=10),
                    dict(t="采样从 1000 步压到 50 步，快 20 倍", size=14.5, space_after=10),
                    dict(t="这几张是从生成结果里挑出来的干净样本。", size=12, color=MUTED, space_after=0),
                ]),
            ],
        ),
        # ---------------- 三个术语展开（紧接最终方案） ----------------
        dict(
            title="刚才那三个词，分别解决什么", notes=P_TERMS_NOTES,
            items=[
                dict(kind="rect", box=(M, 1.28, BODY_W, 1.52), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="text", box=(M + 0.34, 1.38, 3.05, 1.32), anchor=MSO_ANCHOR.MIDDLE, paras=[
                    dict(t="FiLM", size=22, bold=True, color=PRIMARY, space_after=0),
                ]),
                dict(kind="text", box=(M + 3.62, 1.44, 8.55, 1.22), paras=[
                    dict(t="旧做法：所有通道加同一个值 —— 一刀切", size=15, space_after=5),
                    dict(t="FiLM：每个通道有自己的缩放和平移，能分别控制血管、颜色等不同特征", size=15, space_after=5),
                    dict(t="代价很小：参数只多 4.6%", size=13.5, color=MUTED, space_after=0),
                ]),

                dict(kind="rect", box=(M, 3.06, BODY_W, 1.52), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="text", box=(M + 0.34, 3.16, 3.05, 1.32), anchor=MSO_ANCHOR.MIDDLE, paras=[
                    dict(t="L1", size=22, bold=True, color=PRIMARY, space_after=0),
                ]),
                dict(kind="text", box=(M + 3.62, 3.22, 8.55, 1.22), paras=[
                    dict(runs=[("MSE：(x − ", {"name": MATH}), (XHAT, {"name": MATH}), (")", {"name": MATH}),
                               ("2", {"name": MATH, "baseline": 30000}),
                               ("　　　L1：|x − ", {"name": MATH}), (XHAT, {"name": MATH}),
                               ("|", {"name": MATH})], size=16, space_after=5),
                    dict(t="平方让大误差的代价翻倍涨 —— 模型为「不出大错」宁可选中庸", size=15, space_after=5),
                    dict(t="L1 按比例惩罚，不歧视只占 5% 的亮病灶", size=13.5, color=MUTED, space_after=0),
                ]),

                dict(kind="rect", box=(M, 4.84, BODY_W, 1.52), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="text", box=(M + 0.34, 4.94, 3.05, 1.32), anchor=MSO_ANCHOR.MIDDLE, paras=[
                    dict(t="LPIPS", size=22, bold=True, color=PRIMARY, space_after=0),
                ]),
                dict(kind="text", box=(M + 3.62, 5.00, 8.55, 1.22), paras=[
                    dict(t="不比像素，比的是预训练网络中间层的特征图", size=15, space_after=5),
                    dict(t="血管边缘糊掉 1 个像素：像素损失察觉不到，特征差异却很大", size=15, space_after=5),
                    dict(t="权重 0.1，只在低噪声步（t < 200）计算", size=13.5, color=MUTED, space_after=0),
                ]),

                dict(kind="text", box=(M, 6.48, BODY_W, 0.42), paras=[
                    dict(runs=[("总损失 = ", {"name": MATH, "color": MUTED}),
                               ("L1 + 0.1 × LPIPS", {"name": MATH, "color": ACCENT, "bold": True, "size": 19}),
                               ("　（只在低噪声步计算）", {"color": MUTED})],
                         size=16, align=PP_ALIGN.CENTER, space_after=0),
                ]),
            ],
        ),

        # ---------------- P12 评估体系 ----------------
        dict(
            n=12, title="说不清“哪里好”，就不算做完", notes=P12_NOTES,
            items=[
                dict(kind="pic", src=os.path.join(ROOT, "research-report/figures/score_overview.png"),
                     box=(M, 1.28, BODY_W, 4.10)),
                dict(kind="text", box=(M, 5.55, BODY_W, 1.35), paras=[
                    dict(t="最开始只能自己看着打 80 分，被问“依据是什么”就答不上来。", size=16, space_after=8),
                    dict(t="后来搭了两层评估：通用层 11 项 + 眼底专用层（病灶保不保留 / 血管对不对 / 会不会“背下”训练图 / 能不能被一眼识破），最后汇总成六维加权总分。", size=16, color=MUTED, space_after=0),
                ]),
            ],
        ),
        # ---------------- P13 后续进展（ML） ----------------
        dict(
            n=13, title="后续进展：换了个思路", notes=P13_NOTES,
            items=[
                dict(kind="pic", src=A("p13_comparison_grid.png"), box=(10.05, 1.30, 2.75, 5.75)),
                dict(kind="text", box=(M, 1.38, 7.05, 5.60), paras=[
                    dict(t="深度模型受限于数据量，改从“重新组合真实图的零件”入手。", size=17, space_after=16),
                    dict(t="泊松病变重排 —— 把真图的病灶搬到另一张图上", size=16, space_after=4),
                    dict(t="Retinex 光照交换 —— 只换两张图的照明条件", size=16, space_after=16),
                    dict(t="用同一套评估体系量，它们的分数比深度学习高。", size=17, bold=True, color=PRIMARY, space_after=8),
                    dict(t="代价：它们是在搬运真实像素，不会创造新的病灶形态。", size=16, bold=True, color=ACCENT, space_after=0),
                ]),
                *p13_row_labels(),
            ],
        ),
        # ---------------- P14 总结 ----------------
        dict(
            n=14, title="这个暑假我学到的三件事", notes=P14_NOTES,
            items=[
                dict(kind="rect", box=(M, 1.55, BODY_W, 1.20), color=BAND,
                     label="", tcolor=INK, size=1),
                dict(kind="text", box=(M + 0.35, 1.70, BODY_W - 0.7, 0.95), paras=[
                    dict(t="①  数据只有 330 张时，瓶颈是数据，不是模型", size=20, bold=True, color=PRIMARY, space_after=0),
                ]),
                dict(kind="rect", box=(M, 2.95, BODY_W, 1.20), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="text", box=(M + 0.35, 3.10, BODY_W - 0.7, 0.95), paras=[
                    dict(t="②  失败要查到底 —— 两次崩都是能定位的具体原因", size=20, bold=True, color=PRIMARY, space_after=0),
                ]),
                dict(kind="rect", box=(M, 4.35, BODY_W, 1.20), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="text", box=(M + 0.35, 4.50, BODY_W - 0.7, 0.95), paras=[
                    dict(t="③  评估比生成更难，也更值钱", size=20, bold=True, color=ACCENT, space_after=0),
                ]),
                dict(kind="text", box=(M, 5.88, BODY_W, 0.46), paras=[
                    dict(t="还在解决：128×128 下单根血管只有 1~2 像素；约三到四成样本还不够好。", size=15, color=MUTED, space_after=0),
                ]),
                # ↓ 项目链接占位：拿掉方括号、把真实地址粘进去即可
                dict(kind="rect", box=(M, 6.42, BODY_W, 0.50), color=BAND, label="", tcolor=INK, size=1),
                dict(kind="text", box=(M, 6.50, BODY_W, 0.36), paras=[
                    dict(runs=[
                        ("项目地址：", {"color": MUTED}),
                        ("【在此粘贴 GitHub 链接】", {"color": ACCENT, "bold": True}),
                    ], size=15, align=PP_ALIGN.CENTER, space_after=0),
                ]),
            ],
        ),
        # ---------------- P15 备用页 ----------------
        dict(
            n=15, title="备用页：为什么传统方法分数更高", notes=P15_NOTES,
            items=[
                dict(kind="text", box=(M, 1.60, BODY_W, 4.90), paras=[
                    dict(t="被问到再翻这一页。", size=15, color=MUTED, space_after=18),
                    dict(t="泊松 / Retinex 是在搬运真实图的像素 —— 看着真、分数高，是必然的。", size=18, space_after=10),
                    dict(t="生成模型是从零合成整张图 —— 它最大的价值恰恰是能造出训练集里没有的病灶形态。", size=18, space_after=10),
                    dict(t="两者分数可比，但用途根本不同。", size=19, bold=True, color=PRIMARY, space_after=22),
                    dict(t="下一步：把生成的图用于扩充糖尿病视网膜病变分级分类器的 severe 类，验证生成图对下游任务有没有用。", size=16, color=MUTED, space_after=0),
                ]),
            ],
        ),
    ]
    # 页码按列表顺序自动生成 —— 中途插页不会打乱后面所有页的编号
    for i, s in enumerate(spec, 1):
        s["n"] = i
    return spec


# ==========================================================================
# 构建
# ==========================================================================

def new_slide(prs, spec):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if spec.get("title"):
        add_text(slide, (M, TITLE_Y, SW - 2 * M - 0.7, TITLE_H),
                 [dict(t=spec["title"], size=27, bold=True, color=PRIMARY)], line_spacing=1.0)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(M), Inches(RULE_Y),
                                     Inches(BODY_W), Pt(2.5))
        bar.fill.solid()
        bar.fill.fore_color.rgb = PRIMARY
        bar.line.fill.background()
        bar.shadow.inherit = False
    if not spec.get("bleed"):
        add_text(slide, (SW - M - 0.9, 7.06, 0.9, 0.28),
                 [dict(t=str(spec["n"]), size=10, color=RGBColor(0x88, 0x92, 0xA0),
                       align=PP_ALIGN.RIGHT)], line_spacing=1.0)
        add_text(slide, (M, 7.06, 6.0, 0.28),
                 [dict(t="暑期总结 · 眼底图像生成", size=10,
                       color=RGBColor(0x88, 0x92, 0xA0))], line_spacing=1.0)
    return slide


def render_item(slide, item):
    k = item["kind"]
    if k == "pic":
        add_pic(slide, item["src"], item["box"])
    elif k == "text":
        add_text(slide, item["box"], item["paras"],
                 anchor=item.get("anchor", MSO_ANCHOR.TOP))
    elif k == "formula":
        # 公式统一走 Cambria（微软雅黑缺 x̂ 字形）；注解文字仍用微软雅黑
        runs = [(t, {"name": MATH, **o}) for t, o in item["runs"]]
        add_text(slide, item["box"], [dict(
            runs=runs,
            size=item.get("size", 22),
            color=item.get("color", PRIMARY),
            align=item.get("align", PP_ALIGN.CENTER),
            space_after=0,
        )], anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0)
    elif k == "caption":
        x, y, w, h = item["box"]
        add_text(slide, (x, y, w, h),
                 [dict(t=item["t"], size=11.5, color=RGBColor(0x88, 0x92, 0xA0))],
                 line_spacing=1.0)
    elif k == "rect":
        sh = add_rect(slide, item["box"], item["color"])
        if item.get("label"):
            tf = sh.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = item["label"]
            set_run_font(r, size=item.get("size", 16), bold=True,
                         color=item.get("tcolor", INK))
    elif k == "rowlabel":
        x, y, w, h = item["box"]
        add_text(slide, (x, y, w, h),
                 [dict(t=item["t"], size=12.5, bold=item.get("bold", False),
                       color=item.get("color", INK), align=PP_ALIGN.RIGHT)],
                 anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0)
    elif k == "band":
        x, y, w, h = item["box"]
        sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                    Inches(w), Inches(h))
        sh.fill.solid()
        sh.fill.fore_color.rgb = item["color"]
        sh.line.fill.background()
        sh.shadow.inherit = False
        tf = sh.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.12)
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = item["label"]
        set_run_font(r, size=item.get("size", 13), bold=False,
                     color=item.get("tcolor", WHITE))
    else:
        raise ValueError(f"未知元素类型 {k}")


def build():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    spec = slides_spec()
    for s in spec:
        slide = new_slide(prs, s)
        for item in s["items"]:
            render_item(slide, item)
        set_notes(slide, s["notes"])
    prs.save(OUT_PPTX)
    return spec


# ==========================================================================
# 自检
# ==========================================================================

def selfcheck(spec):
    prs = Presentation(OUT_PPTX)
    SWE, SHE, IN = prs.slide_width, prs.slide_height, 914400
    errs, warns = [], []
    expected_pics = {s["n"]: sum(1 for i in s["items"] if i["kind"] == "pic") for s in spec}
    bleed = {s["n"] for s in spec if s.get("bleed")}   # 满幅页允许图文相压

    for i, slide in enumerate(prs.slides, 1):
        npic = 0
        boxes = []
        for sh in slide.shapes:
            if sh.left is None:
                continue
            L, T, W, H = sh.left, sh.top, sh.width, sh.height
            if W <= 0 or H <= 0:
                errs.append(f"P{i} '{sh.name}' 尺寸非法")
                continue
            if L < 0 or T < 0 or L + W > SWE or T + H > SHE:
                if i != 1:      # 首页满幅底图为有意铺满
                    errs.append(
                        f"P{i} '{sh.name}' 越界 "
                        f"({L/IN:.2f},{T/IN:.2f},{W/IN:.2f},{H/IN:.2f})"
                    )
            if sh.shape_type == 13:
                npic += 1
                if not sh.image.blob:
                    errs.append(f"P{i} '{sh.name}' 图片未嵌入")
                iw, ih = sh.image.size
                ar_src, ar_box = iw / ih, W / H
                if abs(ar_src - ar_box) / ar_src > 0.01:
                    errs.append(
                        f"P{i} '{sh.name}' 被拉伸变形：源 {ar_src:.3f} vs 框 {ar_box:.3f}"
                    )
                dpi = iw / (W / IN)
                if dpi < 90:
                    warns.append(f"P{i} 图 {os.path.basename(sh.name)} 仅 {dpi:.0f} DPI")
                boxes.append((L, T, W, H, sh.name))
            elif sh.has_text_frame and sh.text_frame.text.strip():
                boxes.append((L, T, W, H, sh.name))
        if npic != expected_pics[i]:
            errs.append(f"P{i} 图片数 {npic} != 设计 {expected_pics[i]}")
        if not slide.has_notes_slide or not slide.notes_slide.notes_text_frame.text.strip():
            errs.append(f"P{i} 缺讲稿")
        # 图片之间是否互相重叠（封面上标题压在底图上是有意为之）
        if i in bleed:
            continue
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ax, ay, aw, ah, an = boxes[a]
                bx, by, bw, bh, bn = boxes[b]
                ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                oy = max(0, min(ay + ah, by + bh) - max(ay, by))
                if ox * oy > 0.12 * min(aw * ah, bw * bh):
                    warns.append(f"P{i} '{an}' 与 '{bn}' 明显重叠")

    total_sec = sum(len(s["notes"]) for s in spec) / 4.0
    print(f"\n页数 {len(prs.slides)}  讲稿共 {sum(len(s['notes']) for s in spec)} 字 "
          f"≈ {total_sec:.0f} 秒")
    print(f"文件 {os.path.getsize(OUT_PPTX)/1e6:.1f} MB")
    for e in errs:
        print("  [错误]", e)
    for w in warns:
        print("  [提醒]", w)
    if not errs:
        print("  结构自检通过：无越界、无变形、图片数与设计一致、讲稿齐全。")
    return not errs


def make_preview():
    """用同一份 pptx 重新画成 PNG —— 没有 PowerPoint 也能肉眼验证版面。"""
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    prs = Presentation(OUT_PPTX)
    IN = 914400.0
    scale = 1400 / (prs.slide_width / IN)
    f = ImageFont.truetype(FONT_FILE, 15)
    made = []
    for i, slide in enumerate(prs.slides, 1):
        canvas = Image.new("RGB", (1400, int(prs.slide_height / IN * scale)), "white")
        d = ImageDraw.Draw(canvas)
        for sh in slide.shapes:
            x, y = sh.left / IN * scale, sh.top / IN * scale
            w, h = sh.width / IN * scale, sh.height / IN * scale
            if sh.shape_type == 13:
                im = Image.open(io.BytesIO(sh.image.blob)).convert("RGB")
                canvas.paste(im.resize((max(1, int(w)), max(1, int(h))), Image.LANCZOS),
                             (int(x), int(y)))
            elif sh.has_text_frame and sh.text_frame.text.strip():
                d.rectangle([x, y, x + w, y + h], outline=(205, 210, 216))
                txt = sh.text_frame.text.replace("\n", " / ")[:42]
                d.text((x + 2, y + 2), txt, fill=(20, 20, 20), font=f)
            else:
                try:
                    c = sh.fill.fore_color.rgb
                    d.rectangle([x, y, x + w, y + h], fill=(c[0], c[1], c[2]))
                except Exception:
                    d.rectangle([x, y, x + w, y + h], outline=(170, 170, 170))
        d.text((6, 6), f"P{i}", fill=(230, 40, 40), font=f)
        out = os.path.join(PREVIEW_DIR, f"p{i:02d}.png")
        canvas.save(out)
        made.append(out)
    print(f"  预览图 {len(made)} 张 → {os.path.relpath(PREVIEW_DIR, ROOT)}")
    return made


def main() -> int:
    if not os.path.exists(FONT_FILE):
        print(f"错误：找不到中文字体 {FONT_FILE}", file=sys.stderr)
        return 1
    missing = [
        i["src"] for s in slides_spec() for i in s["items"]
        if i["kind"] == "pic" and not os.path.exists(i["src"])
    ]
    if missing:
        print("错误：以下素材不存在，请先跑 _prep_ppt_assets.py：", file=sys.stderr)
        for m in missing:
            print("   ", os.path.relpath(m, ROOT), file=sys.stderr)
        return 1

    print(f"生成 {os.path.relpath(OUT_PPTX, ROOT)}")
    spec = build()
    ok = selfcheck(spec)
    make_preview()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
