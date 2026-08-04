# -*- coding: utf-8 -*-
"""生成 REPORT_ML.docx —— 传统机器学习方法生成眼底彩照（Phase C2）。

数据源：eval_data/ 下各模型 JSON（_metrics.json + _fundus_metrics.json + _scores.json）+ 对比图。
参考：REPORT（原版）_with_metrics.docx 的工程叙述风格 + _insert_metrics.py 的表格样式。
运行：python _build_report_ml.py（在 research-report/ 下），输出 REPORT_ML.docx。
不修改任何已有文件。
"""
import json
import os

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EVAL = os.path.join(ROOT, "eval_data")
ML_FIG = os.path.join(ROOT, "generate_project", "machine_learning", "report", "figures",
                      "trad-ml-vs-deep_comparison.png")
DST = os.path.join(HERE, "REPORT_ML.docx")

# ---------------- 数据加载 ----------------
def load(name, suffix):
    p = os.path.join(EVAL, f"{name}{suffix}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

GM = {m: load(m, "_metrics") for m in ["pca", "gmm", "patch"]}
FM = {m: load(m, "_fundus_metrics") for m in ["pca", "gmm", "patch"]}
SCORES = load("_scores", "")

def score_of(model):
    return SCORES.get("scores", {}).get(model)

def dims_of(model):
    return SCORES.get("dims_scores", {}).get(model, {})

# 深度最佳对照（报告横向对比用）
FID_DEEP = load("film_l1lpips", "_metrics").get("fid", 178.8)

# ---------------- docx 工具 ----------------
doc = Document()

# 中文默认字体（正文等线，标题微软雅黑）
def set_zh(style, zh, en=None, size=None, bold=None, color=None):
    style.font.name = en or zh
    style._element.rPr.rFonts.set(qn("w:eastAsia"), zh)
    if size: style.font.size = Pt(size)
    if bold is not None: style.font.bold = bold
    if color: style.font.color.rgb = RGBColor(*color)

set_zh(doc.styles["Normal"], "等线", "Calibri", 11)
for s in ["Title", "Heading 1", "Heading 2", "Heading 3", "Heading 4"]:
    set_zh(doc.styles[s], "微软雅黑", "Microsoft YaHei")

def H1(text): doc.add_heading(text, level=1)
def H2(text): doc.add_heading(text, level=2)
def H3(text): doc.add_heading(text, level=3)

def P(text, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix + "："); r.bold = True
    p.add_run(text)
    return p

def CODE(text):
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Consolas"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        r.font.size = Pt(9)
        p.paragraph_format.left_indent = Inches(0.3)
    return

def IMG(path, width_in=6.0, caption=None):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width_in))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in cap.runs: r.font.size = Pt(9)
    else:
        P(f"[图缺失: {path}]")

def make_table(header, rows, widths=None, font_size=9):
    t = doc.add_table(rows=1, cols=len(header), style="Table Grid")
    if widths:
        t.autofit = False
        total = sum(widths)
        tblPr = t._tbl.tblPr
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(total)); tw.set(qn("w:type"), "dxa")
        tblPr.append(tw)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        grid = t._tbl.find(qn("w:tblGrid"))
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(w))
        for j, w in enumerate(widths):
            for cell in t.columns[j].cells:
                tcPr = cell._tc.get_or_add_tcPr()
                tcw = OxmlElement("w:tcW"); tcw.set(qn("w:w"), str(w)); tcw.set(qn("w:type"), "dxa")
                tcPr.append(tcw)
    for j, h in enumerate(header):
        run = t.rows[0].cells[j].paragraphs[0].add_run(h)
        run.bold = True; run.font.size = Pt(font_size)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            run = cells[j].paragraphs[0].add_run(str(v))
            run.font.size = Pt(font_size)
    return t

def BULLET(text):
    p = doc.add_paragraph(style="List Bullet"); p.add_run(text); return p

def sixdim_table(model_label, model_key):
    """单个方法 + 深度最佳的六维评分对比表。"""
    labels = ["D1 病灶保留", "D2 抗识破+分布", "D3 多样性/质量", "D4 血管结构", "D5 颜色", "D6 记忆风险"]
    d = dims_of(model_key)
    d0 = dims_of("film_l1lpips")
    rows = [[l, f"{d0.get(l, float('nan')):.2f}", f"{d.get(l, float('nan')):.2f}"] for l in labels]
    rows.append(["总分(0-100)", f"{score_of('film_l1lpips'):.1f}", f"{score_of(model_key):.1f}"])
    make_table(["评分维度", "FiLM+LPIPS(深度最佳)", model_label], rows,
               widths=[4200, 3000, 3000])

# ==================================================================
# 封面
# ==================================================================
t = doc.add_paragraph("REPORT_ML：传统机器学习方法生成眼底彩照（Phase C2）")
t.style = doc.styles["Title"]
sub = doc.add_paragraph("小样本重度 DR 图生成的边界与组合潜力 —— 面向小白")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
meta = doc.add_paragraph("2026-08-03 ｜ 数据：330 张 KW-IV 重度眼底彩照 ｜ 方法：PCA / GMM / 补丁拼接（基线）+ 组合管线（后续）")
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.runs[0].font.size = Pt(10)

# ==================================================================
# 1. 数据集
# ==================================================================
H1("1. 数据集")
P("本报告用 330 张严重糖尿病视网膜病变（DR，Keith-Wagener IV 级）眼底彩照，系统试了三类传统机器学习方法：PCA 线性重建、GMM 混合采样、补丁拼接。它们都不含神经网络、不反向传播、不梯度训练——是真正的传统 ML（非深度学习）方法，回应老师 Bug 2 里\"方法不限于深度学习\"的要求。")
P("这 330 张图与 Phase A/B 深度方法用同一批数据、同一种口径：统一缩放到 128×128 RGB，评估时以 330 张真实图为对照。")
P("一个对传统 ML 特别有利、也特别不利的事实：KW IV 级意味着每张图都病灶密集——出血、渗出、微动脉瘤到处都是。有利的一面是它自带一座\"病变模板库\"（后面泊松病变重排就靠它）；不利的一面是\"重度\"让数据分布高度非高斯，正好打在参数化模型最弱的位置上。")
make_table(["项目", "数值", "说明"], [
    ["样本数", "330", "全部 KW-IV 重度 DR，与深度方法同源"],
    ["分辨率", "128×128 RGB", "统一缩放口径（与 Phase A/B 一致）"],
    ["评估对照", "330 张真实图", "eval_data/real/"],
    ["生成量", "300 张/方法", "seed=42，sample_0000~0299"],
], widths=[2500, 2200, 5300])

# ==================================================================
# 2. 评估指标与评分标准
# ==================================================================
H1("2. 评估指标与评分标准")
H2("2.1 为什么用同一套评估体系")
P("Phase A/B 已经为 6 个深度方法建好了一套评估体系（通用层 + 专用层 + 六维门控综合评分）。传统 ML 方法复用同一套，才能和深度方法在同一个标尺上横向对比——这正是\"补丁拼接 42.1 分 vs FiLM 扩散 70.9 分\"这类结论的前提。否则各说各话，报告没有说服力。")

H2("2.2 通用层指标（对标文献）")
P("10 项通用指标，方向：FID/KID/MMD/MS-SSIM/LPIPS 越低越好；IS/Precision/Recall/Density/Coverage 越高越好；1-NN 越接近 0.5 越好。FID/KID 用 ImageNet 预训练 Inception 特征，128×128 输入。")
make_table(["指标", "含义（小白版）", "方向"], [
    ["FID", "生成图分布离真实图有多远", "↓ 低"],
    ["KID", "FID 的无偏版本，小样本更稳", "↓ 低"],
    ["1-NN", "真假混合最近邻能否分开（0.5=分不开）", "→ 0.5"],
    ["MS-SSIM", "内部多样性（两两相似度）", "↓ 低"],
    ["LPIPS-NN", "最近邻感知距离（越大越不复制）", "↓ 低"],
    ["IS", "类内清晰+类间多样", "↑ 高"],
], widths=[1500, 6200, 1300])

H2("2.3 专用层自设计指标（回应 Bug 1 的\"自设计\"）")
P("针对眼底图设计：病灶保留（出血/渗出）用 Wasserstein 距离 + 保留率（≈1 理想，>1 偏多 <1 偏少）；血管结构用占比分布距离；抗识破用 C2ST（训练小 CNN 区分真假，AUC 越低越难被识破）；记忆风险用最近邻 SSIM（>0.85 占比 = 接近复制比例）；图像质量用 BRISQUE（真实眼底 = 3.69，仅参考）。")

H2("2.4 六维门控综合评分（0-100）")
P("六个维度加权：D1 病灶保留 0.30、D2 抗识破+分布 0.25、D3 多样性/质量 0.20、D4 血管结构 0.10、D5 颜色 0.08、D6 记忆风险 0.07。门控：D2 是\"现实主义\"闸门，D2 得分低于阈值则总分被扣（防止\"糊但指标好看\"的假象）。")

H2("2.5 复制检测三件套（质检门控）")
P("330 张小样本处于记忆/复制高风险区（文献显示 GLO 在 128 张就开始记忆训练集）。每个方法生成后必须过三件套：(1) 最近邻 NN-SSIM/LPIPS 距离分布；(2) 全图复制率；(3) 补丁重复率。任一超标 → 停。")

H2("2.6 人眼评分说明")
P("综合评分是客观指标，人眼评分（人类对\"像不像眼底图\"的主观打分）对 ML 方法标注为【待人工评估】——本报告不虚造人工分。效果图 + 客观分已在各方法章给出，人眼分待人工复核后补录。")

H2("2.7 ⚠️ 踩坑教训：C2ST 必须跑")
P("本报告早期按\"传统 ML 无条件图跳过 C2ST\"执行，结果补丁拼接打出 62.9 分的伪高分——逼近深度最佳 70.9。排查发现根因是评分口径不均：深度模型的 D2 维度含 C2ST（权重 0.40），而 ML 基线没有，导致 D2 在不同指标集上对比。补跑 C2ST 后补丁回落到 42.1，GMM 升到 40.2、PCA 升到 20.0，全部同口径。这条教训已写进执行计划：C2ST 必跑，不能跳过。")

# ==================================================================
# 3. 传统 ML 方法谱系与可行性
# ==================================================================
H1("3. 传统 ML 方法谱系与可行性")
H2("3.1 一句话：18 法分三档")
P("文献调研盘点了 18 种传统 ML 图像生成方法，按\"在 330 张重度眼底图上可行的程度\"分三档：")
make_table(["档位", "代表方法", "在眼底图上的判断"], [
    ["高", "Image Quilting 拼布、眼底补丁合成（Fiorini 2014 / Magnusson 2021）、稀疏字典、Poisson 融合、颜色/直方图匹配", "非参数\"重排真实块\"路线，保纹理真实，是唯一能保\"相似\"的路线"],
    ["中", "COCA（Copula PCA 修正）、GMM 紧凑表示、AAM/SSM 结构插值、示例式修补", "有改进空间，但离直接生成仍有距离"],
    ["低", "纯 PCA/PPCA/FA、KDE、FRAME、纯 MRF", "参数化/密度模型，必败于非高斯+强结构（病灶被平均掉）"],
], widths=[1100, 5200, 3700])

H2("3.2 核心规律：分布假设决定能力边界")
P("读完 18 法，最深的一条规律是：传统 ML 生成器的能力边界，由它\"假设的数据分布形状\"决定。")
BULLET("参数化模型（PCA/GMM/密度）假设数据是高斯/线性的，而眼底图是强非高斯（黑背景/亮视盘/暗病灶）+ 强结构（血管走向）。线性假设把一切拉向\"平均脸\"，病灶被平均掉——这是它们必败的根源。")
BULLET("非参数补丁法不假设全局分布，直接重排真实纹理块，因此天然保\"相似\"；但它不产新结构，缺全局布局。")
P("结论：单一传统 ML 方法不行，但\"重排真实块 + 另一层传统方法补全局结构\"的组合管线可行——这正是本报告后续方向的出发点。")

H2("3.3 为什么选这三个做基线")
P("三个基线恰好覆盖方法谱系的两端和一个特例：PCA（参数化线性代表）、GMM（参数化密度代表）、补丁拼接（非参数/无模型特例，连参数都没有）。把它们跑出来，就同时验证了\"参数化必败\"和\"非参数保相似缺结构\"两条规律。")

H2("3.4 生成 vs 增广：C2 的价值定位")
P("文献证据（03 篇）显示：经典增广（旋转/mixup）对分类器的判别力提升，普遍不输甚至超过 GAN/扩散生成图扩充。因此本报告不说\"传统 ML 生成比增广强\"，而说：传统 ML 组合能产出**经典增广做不到的新病灶组合**（旋转/mixup 不产生新病灶布局，泊松病变重排能）。C2 的价值锚定在\"新组合\"而非\"更强的判别力\"。")

H2("3.5 五方法效果总览")
P("真实原图、深度最佳、三个传统 ML 基线的并排对比（每列一张样本，同索引）：一眼可辨——真实图清晰，扩散最接近真实，PCA/GMM 过平滑成\"模糊团\"，补丁是清晰的四块拼贴。")
IMG(ML_FIG, 6.3, "图 1：真实 vs 深度最佳(FiLM 扩散) vs 三个传统 ML 基线生成效果对比")

# ==================================================================
# 4. PCA 线性重建（基线 1）
# ==================================================================
H1("4. PCA 线性重建（基线 1）")

H2("4.1 是什么")
P("PCA（主成分分析）是最经典的传统 ML 线性方法：把 330 张图各自压平成 49152 维向量，找 64 个\"最主要的方向\"（主成分）表达全数据。生成 = 在潜空间随机采样，再反投影回像素。", "它")
P("一个小白最容易误解的点：PCA 连\"训练轮数\"都没有。它是解析解——一次协方差矩阵的特征分解，数学上直接得到全局最优，不存在\"多训几轮会更好\"。", "特殊点")

H2("4.2 怎么做（略写）")
CODE("python scripts/pca_gen.py --num_images 300 --seed 42\n# PCA 拟合 64 维（保留 96% 方差）→ 潜空间随机采样 → 反投影 → clip 0-255")

H2("4.3 为什么结果必然过平滑")
P("PCA 假设 330 张图躺在一个 64 维线性子空间里。压缩到 64 维时，锐利的病灶边缘、细小血管这类\"高频细节\"被当作噪声丢掉了——这是结构性信息损失，无论怎么算都找不回来。生成的图本质是\"所有图的平均态\"：像一张模糊的合成眼底，谁都不像。")

H2("4.4 结果")
P("总分 20.0 分，FID 297.1（全组最差，深度最佳 178.8）。1-NN=1.000（完美可分），IS=1.58（多样性几乎为零）。病灶保留率：出血 1.57（偏多）、渗出 0.56（偏少）——连病灶比例都没对齐。记忆检测通过（接近复制占比 0）。人眼评分【待人工评估】。")
sixdim_table("PCA(基线1)", "pca")
IMG(os.path.join(EVAL, "pca", "singles", "sample_0000.png"), 2.2, "PCA 生成示例 sample_0000（注意整体模糊、病灶消失）")

H2("4.5 致命缺点")
BULLET("过平滑：线性假设把病灶平均掉，没有任何锐利细节")
BULLET("无新病灶形态：所有输出只是\"平均脸\"的随机抖动")
BULLET("无全局结构：血管走向、视盘位置全靠碰运气")

H2("4.6 下一步")
P("PCA 的失败证实了\"参数化线性方法不可行\"，但它的潜空间仍可用——COCA（高斯 copula 修正）能改善非高斯伪影；更重要的是，PCA 可作为结构层（血管/布局）的紧凑表示，与纹理层组合。单个不行，组合是出路。")

# ==================================================================
# 5. GMM 混合采样（基线 2）
# ==================================================================
H1("5. GMM 混合采样（基线 2）")

H2("5.1 是什么")
P("GMM（高斯混合模型）是传统 ML 概率密度估计：用 16 个高斯分量去拟合降维后的数据分布，生成 = 从拟合好的混合分布里采样。", "它")
P("GMM 有训练迭代（EM 算法），但实测给 10/100/1000 次迭代上限，它 2 次就收敛，BIC 完全一样——不是\"训练不够\"。过平滑是生成机制的属性，不是收敛问题。", "特殊点")

H2("5.2 怎么做（略写）")
CODE("python scripts/gmm_gen.py --num_images 300 --seed 42\n# PCA 降维 64 → GMM(16 分量) 拟合 → 混合分布采样 → 反投影")

H2("5.3 为什么结果必然过平滑")
P("GMM 的每个分量是一个高斯分布，而高斯分布天生光滑。从混合高斯里采样的图，尖锐的出血边缘、血管分叉根本不在分布里——模型想表达也表达不出来。这比 PCA 强的地方是它能刻画多模态（比如\"两种血管走向\"各是一个高斯峰），但病灶细节依然被抹平。")

H2("5.4 结果")
P("总分 40.2 分，FID 233.8，1-NN=1.000。比 PCA 好：能出大致形状和轮廓，病灶保留率也更接近 1（出血 1.43、渗出 0.71）。但病灶仍模糊，放大即露馅。记忆检测通过。人眼评分【待人工评估】。")
sixdim_table("GMM(基线2)", "gmm")
IMG(os.path.join(EVAL, "gmm", "singles", "sample_0000.png"), 2.2, "GMM 生成示例 sample_0000（轮廓接近真实但细节仍糊）")

H2("5.5 致命缺点")
BULLET("过平滑：高斯分量的光滑性决定了病灶边缘模糊")
BULLET("无新病灶形态：采样只会在已有分布附近抖动，造不出新的病变组合")
BULLET("对 330 张重度图，多模态优势发挥有限——分布本身太集中")

H2("5.6 下一步")
P("GMM 比 PCA 稍好，但天花板相同。它的价值同样是\"层次之一\"：GMM 可以在结构层对血管/病灶布局建模，与纹理层组合。组合管线再次是答案。")

# ==================================================================
# 6. 补丁拼接（基线 3）
# ==================================================================
H1("6. 补丁拼接（基线 3）")

H2("6.1 是什么")
P("补丁拼接是无模型方法：从 330 张真实图里随机裁 64×64 块，4 块拼成一张 2×2 马赛克。它连参数都没有，是传统 ML 谱系里最纯粹的\"数据重排\"特例，也是 Image Quilting（Efros & Freeman 2001）的最朴素版本。", "它")

H2("6.2 怎么做（略写）")
CODE("python scripts/patch_gen.py --num_images 300 --seed 42\n# 随机选图 + 随机起点裁 64×64 真实块 → 4 块填入 2×2 → 保存")

H2("6.3 为什么它\"保相似但不生成\"")
P("补丁法不假设全局分布，不压缩信息，只重组真实像素——所以纹理、颜色、病灶比例都\"真\"。但它不做结构，接缝处也不对齐，输出是\"清晰但机械的拼贴\"。这正是文献里说的：非参数补丁法保\"相似\"、缺\"结构\"。")

H2("6.4 结果（含门控发现）")
P("总分 42.1 分，FID 159.2（三个 ML 里最低，甚至比部分深度模型还低），1-NN=0.995。病灶保留率最接近 1（出血 0.94、渗出 1.16），血管 Wasserstein 最小——因为像素本来就是真的。C2ST=1.000：真实像素拼接被小 CNN 一眼识破。记忆检测通过（接近复制占比 0）。人眼评分【待人工评估】。")
sixdim_table("补丁拼接(基线3)", "patch")
P("本方法最早得 62.9 分的伪高分，后经排查确认是 C2ST 缺失导致的口径不均（见 2.7），补跑后回落到 42.1。这个\"伪高分事件\"本身是宝贵的：它暴露了评分体系的一个盲区——凡是用真实像素的方法，病灶/血管/颜色三维必然高分。报告中如实记录，作为评估体系的改进依据。")
IMG(os.path.join(EVAL, "patch", "singles", "sample_0000.png"), 2.2, "补丁拼接示例 sample_0000（清晰但 4 块拼贴、接缝可见）")

H2("6.5 致命缺点")
BULLET("无意义机械重排：每张都不同，但没有医学意义的新组合")
BULLET("接缝伪影：块间不做平滑/对齐")
BULLET("局部复制风险：每一像素都来自真实图，需复制检测盯防")

H2("6.6 下一步")
P("补丁法证明了\"非参数保相似\"——它能保住纹理真实。它缺的\"全局结构\"，恰好可以由另一层传统方法补：血管骨架生成 + 病灶布局规划。这就是\"组合管线\"的思路：泊松病变重排（保留真实的病灶、加上宿主图结构）和 Retinex 光照交换（保留结构、交换光照），都比纯拼接多一层语义。")

# ==================================================================
# 7. 高价值组合：后续方向
# ==================================================================
H1("7. 高价值组合：后续方向【待实验】")

P("三个基线的结论指向同一个方向：单一传统 ML 不行，但\"组合\"可行。文献里最有支撑的两条组合（均有眼底专版验证）如下，本报告规划为下一阶段实验，尚未跑，不虚写结果。")

H2("7.1 泊松病变重排（文献基础：强）")
P("做法：330 张全是重度 DR = 一座病变模板库。从图 A 抠一簇渗出、图 B 抠一块出血，用泊松编辑无缝贴到图 D 上，产出\"新病变组合\"的严重 DR 图。")
P("为什么有用：泊松编辑只保留插入区域的梯度、用宿主图边界条件求解，所以贴上去的病灶无缝融合、不破坏宿主图。它生成的是经典增广永远做不到的东西——新的病灶组合。（Yu 2021 BOE：DR 筛查优于过采样/裁剪/旋转）", "机制")

H2("7.2 Retinex 光照交换（文献基础：强）")
P("做法：每张图分解为光照 L × 反射 R，跨图交换 L、保留各自 R，重乘出新图。")
P("为什么有用：反射 R 承载解剖结构（血管/视盘/病灶），光照 L 承载采集明暗色温。交换 L 只改\"看起来的调子\"，诊断语义几乎不变——安全的多样性放大器。（Zhang 2022 CBM：分割提升 9.6% Dice）", "机制")

H2("7.3 可选：血管骨架 + 纹理拼布管线")
P("用传统方法（ASM/Bonaldi 或过程化 CCO/DLA）生成全新血管骨架，沿骨架拼真实纹理，inpainting 补缝。结构全新 + 纹理真实，最接近\"真生成\"，但工程量最高。（Fiorini 2014 / Magnusson 2021 已在眼底验证并提升分割 SOTA）")

H2("7.4 为什么这两个组合值得做")
P("深度无条件生成器难以显式控制\"病灶种类/密度/位置\"的组合；而泊松/Retinex 用显式规则做到。这正是本报告的核心论点：传统 ML 组合做深度方法做不到的事，作为数据扩充具有补充价值。")

# ==================================================================
# 8. 结论
# ==================================================================
H1("8. 结论")
H2("8.1 项目回顾")
P("本报告用 330 张严重 DR 眼底彩照，系统试了三类传统 ML 方法（PCA 20.0 / GMM 40.2 / 补丁 42.1），与 Phase A/B 的 6 个深度方法在同一套评估体系下横向对比（深度最佳 FiLM+LPIPS 70.9）。")

H2("8.2 核心发现")
BULLET("纯传统 ML 直接生成不可行：PCA/GMM 的过平滑证实\"参数化方法必败于非高斯+强结构\"；补丁虽得 42.1，但那是真实像素的平凡胜利，不是生成。")
BULLET("评分体系盲区被暴露：补丁伪高分（62.9→42.1）证明\"用真实像素的方法必然在病灶/血管/颜色维度高分\"，跨\"真实重排 vs 生成\"对比时 C2ST 必须跑。")
BULLET("非参数保\"相似\"、参数化保不了：补丁法保纹理真实但缺结构，恰好引出组合管线。")

H2("8.3 C2 的价值定位")
P("本报告的价值不在于\"传统 ML 比深度强\"（文献不支持，也不争），而在于两点：一是完整的方法论对照——证明\"为什么深度生成/组合管线是必要的\"，回应老师 Bug 2；二是传统 ML 组合能产出深度无条件生成器难以显式控制的\"新病灶组合\"，作为数据扩充有补充价值。")

H2("8.4 局限")
BULLET("无真实病灶 mask：病灶保留率基于像素统计估计，非专家标注")
BULLET("330 张小样本：记忆/复制风险高，评估必须患者级划分")
BULLET("128×128 分辨率偏低；BRISQUE 在自然图像上训练，眼底仅参考")

H2("8.5 下一步")
P("阶段 2 执行泊松病变重排 + Retinex 光照交换（各 300 张，同一套评估），验证\"新病灶组合\"假设；然后进入 Phase D 下游分类器验证（TSTR/TRTR 协议）。每步实验结果将按六段式更新到本报告。")

# ==================================================================
# 参考文献
# ==================================================================
H1("参考文献")
refs = [
    "Efros A A, Freeman W T. Image Quilting for Texture Synthesis and Transfer. SIGGRAPH 2001.",
    "Pérez P, Gangnet M, Blake A. Poisson Image Editing. SIGGRAPH 2003.",
    "Fiorini S, et al. STAG: fundus image synthesis by image quilting (视网膜合成). 2014.",
    "Bonaldi L, et al. Automatic generation of synthetic retinal fundus images (ASM 血管树). 2016.",
    "Magnusson J, et al. DeLTA: fundus segmentation with synthetic data (DRIVE/STARE SOTA). 2021.",
    "Yu Z, et al. BOE: Biomedical image synthesis via object-level rearrangement (泊松病变重排). 2021.",
    "Zhang J, et al. CBM: Color-consistent fundus image synthesis with Retinex (光照交换). 2022.",
    "Egger B, et al. COCA: Copula Eigenfaces for face synthesis (Copula PCA). 2016.",
    "Feng Q, et al. Training data memorization in generative models (ICCV 2021).",
    "Ahamed M, et al. Mixup for diabetic retinopathy classification. 2025.",
    "Beinecke J, et al. Data augmentation in medical imaging: when does it help? 2021.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph(f"[{i}] {r}")
    p.paragraph_format.left_indent = Inches(0.3)
    for run in p.runs: run.font.size = Pt(9)

doc.save(DST)
print(f"已生成 {DST}")

# ---- 简单自检 ----
h1 = [p.text for p in doc.paragraphs if p.style.name == "Heading 1"]
print(f"H1 章数: {len(h1)}")
for h in h1: print("  ", h)
print(f"表格数: {len(doc.tables)}  段落数: {len(doc.paragraphs)}")
