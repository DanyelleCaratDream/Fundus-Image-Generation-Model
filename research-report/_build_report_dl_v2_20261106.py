#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重建 REPORT_DL_v2.docx（2026-11-06 全面重写）

- 以旧 REPORT_DL.docx 为模板（保留样式/页面），清空正文后按 Q1-Q4 框架重写；
- 保留 9 张关键旧图 + 2 张新图；
- EX-001~EX-018 原样复制到附录 A；
- 输出 v2，不覆盖原文件。

用法
    python research-report/_build_report_dl_v2_20261106.py
"""
import copy
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "research-report" / "REPORT_DL.docx"
NEW = REPO / "research-report" / "REPORT_DL_v2.docx"
MEDIA = REPO / "research-report" / "_media_dl"
FIG = REPO / "research-report" / "figures"


def blob_by_caption(doc, prefix):
    """找到以 prefix 开头的图注段落，向上回溯取最近的 drawing 图片 blob。"""
    paras = doc.paragraphs
    for i, p in enumerate(paras):
        if p.text.strip().startswith(prefix):
            for j in range(i, max(-1, i - 5), -1):
                blips = paras[j]._element.findall(".//" + qn("a:blip"))
                if blips:
                    rid = blips[0].get(qn("r:embed"))
                    return doc.part.related_parts[rid].blob
    raise RuntimeError(f"图片未找到: {prefix}")


def collect_range(doc, a, b):
    """按段落索引复制 body 子元素（含其间表格/图片），返回 deepcopy 列表。"""
    body = doc.element.body
    start = doc.paragraphs[a]._element
    end = doc.paragraphs[b]._element
    out, on = [], False
    for child in list(body):
        if child is start:
            on = True
        if on:
            out.append(copy.deepcopy(child))
        if child is end:
            break
    return out


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    doc = Document(str(OLD))

    # ---- 1) 先提取图片 ----
    img_specs = {
        "fig_dcgan": "图 5：DCGAN",
        "fig_ddpm230": "图 8：DDPM",
        "fig_cond800": "图 10c：条件扩散",
        "fig_film": "图 13a：FiLM DDPM + L1+LPIPS",
        "fig_cfm_curve": "图 6-3-1",
        "fig_bdomain": "图 6-4-1",
        "fig_same_slot": "图 6-4-2",
        "fig_sr": "图 6-4-3",
        "fig_lesion": "图 6-4-4",
    }
    img_paths = {}
    for name, cap in img_specs.items():
        p = MEDIA / f"{name}.png"
        p.write_bytes(blob_by_caption(doc, cap))
        img_paths[name] = p
        print(f"[img] {name} <- {cap}")
    img_paths["fig_ablation"] = FIG / "fig6_ablation_6methods.png"
    img_paths["fig_misalign"] = (
        REPO / "generate_project/deep_learning/Fundus-FlowMatching/CFM/judgment/weight_misalign_demo.png"
    )

    # ---- 2) 收集附录 A 的 EX 区块（段落索引已验证） ----
    ex_ranges = [(115, 171), (250, 317), (387, 431), (718, 731)]
    appendix = []
    for a, b in ex_ranges:
        blk = collect_range(doc, a, b)
        print(f"[appendix] {a}-{b}: {len(blk)} blocks")
        appendix.extend(blk)

    # ---- 3) 清空正文（保留 sectPr） ----
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)

    # ---- 4) 写作辅助 ----
    def H(text, level=1):
        doc.add_heading(text, level=level)

    def P(text="", bold_lead=None):
        p = doc.add_paragraph()
        if bold_lead:
            r = p.add_run(bold_lead)
            r.bold = True
        p.add_run(text)
        return p

    def B(text):
        doc.add_paragraph(text, style="List Bullet")

    def TABLE(headers, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        hdr = t.rows[0].cells
        for i, h in enumerate(headers):
            hdr[i].text = ""
            run = hdr[i].paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(9)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v))
                run.font.size = Pt(9)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        doc.add_paragraph()
        return t

    def FIGURE(key, caption, width=6.0):
        p = img_paths.get(key)
        if p is None or not Path(p).exists():
            print(f"[WARN] missing figure {key}")
            return
        doc.add_picture(str(p), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)

    # ============================================================
    # 封面 / 摘要
    # ============================================================
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("眼底彩照生成模型研究报告（v2）")
    tr.bold = True
    tr.font.size = Pt(22)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Fundus Image Generation: VAE, GAN, Diffusion, Flow Matching")
    sr.font.size = Pt(12)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = meta.add_run("2026-11-06 重写版 · 新增 2026-10-26 消融与 2026-11-06 自查更正")
    mr.font.size = Pt(9)

    H("摘要", 2)
    P("本报告研究小样本高血压视网膜病变（hypertensive retinopathy, HR）眼底彩照的生成："
      "在严重样本极少的条件下，如何生成整体真实、解剖结构合理、病灶可辨的图像，并为下游分级"
      "提供可用的数据扩充。工作分三段。第一段在 330 张 A 域网络上系统比较 VAE、GAN、Diffusion "
      "三代模型，定位三类失效机理——模糊天花板、判别器记忆、MSE 病灶溶解——并用 L1+LPIPS 组合"
      "损失显著缓解溶解。第二段迁移到合作导师的 HR 数据集（758 张，severe 仅 76 张唯一底图），"
      "改用条件流匹配（conditional flow matching, CFM）做原生 256 生成：FID 121.84，"
      "作者目检 97/100，分布与感知指标明显优于 DDPM 微调线 D3（FID 222.69）。"
      "第三段按 Q1 整体真实性、Q2 结构与病灶保真、Q3 多样性、Q4 下游验证四个问题复核评价体系，"
      "并完成一次系统自查：发现并修复加权损失的 3 倍尺度缺陷与图像-条件增强错位，重算无偏 KID，"
      "审计跨 split 重复（32 对），更正 HR/DR 术语与多项指标口径。")
    P("结论边界：当前证据支持 CFM 在真实性与分布吻合上优于 DDPM 微调线；病灶保真仍受训练数据"
      "信息量与代理指标限制；渗出加权的旧结果因训练缺陷仅作探索性记录，修正实验待重训。"
      "本报告不声称临床可用性、留出泛化能力或方法优越性的因果结论。")

    # ============================================================
    H("1. 研究问题与数据", 1)

    H("1.1 四个研究问题", 2)
    P("生成医学图像不是单一目标：整体像、结构对、病灶在、能用于下游，是四件不同的事。"
      "本报告按四个问题组织证据，而不是把模型按家族各写一章。")
    B("Q1 整体真实性：生成图的颜色、纹理、光照与整体分布是否接近真实眼底？")
    B("Q2 结构与病灶保真：血管走行、视盘位置与大小、出血/渗出等关键特征是否合理保留？")
    B("Q3 多样性：模型是否覆盖了训练集中的真实变化，而不是反复生成相似图？")
    B("Q4 下游验证：经医生认可的前提下，合成图能否为分级模型提供有效补充？")
    P("对应地，本报告把评价拆成两层：通用分布层（FID/KID/C2ST 等）回答“整体像不像”，"
      "医学结构层（血管、视盘、病灶代理指标、医生盲评）回答“医学上对不对”。两层不可互相替代。")

    H("1.2 两域数据", 2)
    TABLE(
        ["数据域", "来源", "规模", "标签/分辨率", "用途"],
        [
            ["A 域", "网络收集的重度眼底彩照", "330 张快照；现存 299（淘汰 31）", "无标签；128px", "早期模型横评"],
            ["B 域", "合作导师 HRSG_4_758（HR，KW I–IV）", "758 张：train 552 / val 135 / test 71", "normal/mild/moderate/severe；256px", "迁移与交付"],
            ["B 域 severe", "同一数据集", "train 152 文件 = 76 唯一 + 76 翻转；val 21；test 13", "severe = KW-IV", "生成器训练/评估"],
            ["生成训练池", "A 域 132 + 老师 train 派生 61", "193 张唯一（combined_sq256）", "无标签；256px", "D3/D4/CFM 训练"],
        ],
        widths=[1.0, 2.2, 2.2, 1.8, 1.2],
    )
    P("术语先钉死：本项目疾病是高血压视网膜病变（HR），分级采用 Keith-Wagener-Barker（KW I–IV），"
      "severe 对应 KW-IV。此前报告与代码中多处误写为糖尿病视网膜病变（DR），已于 2026-11-06 全仓库更正；"
      "DR 仅保留在引用他人工作（DR-GAN、DRIVE、ETDRS）与历史借用 prompt 中。")
    P("severe 类的稀缺是本项目的直接动机：train 里 76 张唯一底图（另 76 张是数据集自带的翻转副本），"
      "test 仅 13 张。任何以 severe 为中心的指标都必须在这个样本量下解读——1 张图约等于 7.7 个百分点的召回波动。")

    H("1.3 数据卫生与审计", 2)
    P("生成器训练池从 239 → 193：去掉池内重复与跨 split 泄漏文件。2026-11-06 对老师数据集做了"
      "一次穷举审计（123,297 对跨 split 组合，128px 中心圆盘）：严格分层得到 32 对候选"
      "（5 对 exact、27 对 near，MSE ≤ 1e-3），3 对已知像素级重复全部命中——"
      "80024(val)≡80052(train)、80029(val)≡80038(train)、80028(test)≡80055(train)；"
      "另有 16788/16796(train)≡16794(val)。dHash 未参与判定：本数据上 dhash=0 的 93 对里"
      "MSE 最高达 0.049，单独使用会误报上万对。")
    P("影响：老师官方 split 本身含跨 split 同图，下游分类基线 QWK 0.747 含未知幅度的乐观偏差。"
      "建议按眼/序列重新划分；在重划分之前，本报告不声称下游指标的干净泛化。")
    P("另一个必须声明的口径：B 域 FID 的参照集是老师 train/severe 的 72 张唯一底图，"
      "与生成器训练池同源（63/72 内容相同、61 张像素级一致）。因此 FID 衡量的是训练分布拟合，"
      "不是留出泛化；A 域（330 张）与 B 域（72 张）参照不同，禁止跨章比较 FID/KID/C2ST。")

    H("1.4 评估口径（摘要）", 2)
    B("通用层：FID、KID、C2ST AUC、BRISQUE、MS-SSIM、LPIPS。")
    B("医学层：血管 Dice/覆盖率、疑似出血/渗出像素占比（颜色阈值代理，未验证）、记忆检测。")
    B("KID 于 2026-11-06 改为无偏子集自助（100 子集，返回 mean±std）；旧值保留为 kid_old，二者不可混比。")
    B("C2ST 理想值为 0.5；本报告所有方法仍在 0.836–0.944，只能作相对排序。")
    B("病灶指标是颜色阈值启发式，未经像素级医学标注验证；只作相对代理，不作绝对病灶保真结论。")

    # ============================================================
    H("2. 方法演进", 1)

    H("2.1 从 VAE 到扩散（压缩回顾）", 2)
    P("早期路线是“逐家族试错”。VAE 在 330 张上稳定收敛，但输出先天模糊——逐像素 MSE 的最优解是"
      "条件均值，细节被平均掉；把 latent 维度与容量翻倍、训练从 800 轮延到 1200 轮，肉眼仍无差别。"
      "GAN 路线（DCGAN、WGAN-GP 标准版/削弱版、StyleGAN2-ADA）在小样本下撞上判别器过拟合："
      "标准版 WGAN-GP 两个 epoch 内 Critic 分数从 151.68 爆到 −398.89；削弱版训练 2770 轮后"
      "生成图从第 50 轮起几乎不变；StyleGAN2-ADA 因 CUDA 自定义算子与 PyTorch 2.11/Windows 不兼容未能运行。")
    P("转向扩散后，DDPM 稳定但细节不足；条件扩散（血管 mask 加性融合）与 Palette（通道拼接）"
      "都出现“亮病灶随训练被拉暗”的溶解效应。最终有效的是 FiLM 条件 UNet + L1 + LPIPS 的组合。")

    H("2.2 FiLM 条件 UNet 与 L1 + LPIPS", 2)
    P("架构沿用 DDPM 的 UNet（base_dim 128，dim_mults 1/2/3/4，注意力在第 2 层），血管骨架通过"
      "FiLM 调制注入每个 ResBlock；EMA decay 0.9999。损失改为 L1（对离群值不敏感）并在低噪声段"
      "（t<200）加入 LPIPS 感知损失（权重 0.1），另加颜色均值正则（0.01）。这套组合在 A 域把"
      "出血保留的代理指标从 0.254 提升到 1.137（4.5×），FID 200.2→178.8；在 B 域微调得到 D3。")

    H("2.3 条件流匹配（CFM）", 2)
    P("CFM 把生成建模为噪声到数据的线性最优传输路径：x_t=(1−t)x_0+t·x_1，网络回归速度 "
      "v=x_1−x_0，采样用 Euler ODE。与 DDPM 相比，路径更直、目标更稳；架构与参数可复用，"
      "只需重学输出头。实现上使用独立耦合（非 minibatch OT），未做 reflow。")
    P("迁移时发现一个隐性冲突：DDPM 的离散下标 0=干净、999=噪声，而 CFM 的 t=0=噪声、t=1=数据。"
      "原实现把 CFM 的 t 直接映射到 0..999，等于让模型“倒着学”时间条件（首轮 loss 0.904，"
      "接近预测零速度的 E|v|）。修复为 idx=(1−t)·T 后，FID 从 178.9 降到 151.8（−15%）。")

    H("2.4 条件扩展", 2)
    B("文本条件：零初始化 text_proj 把 severe 文本嵌入加到时间嵌入，不破坏预训练（加载后 maxdiff=0.0）。")
    B("检索条件：类内 k=2 最近邻 + Dirichlet 重心采样；C 路用 15 维手工特征，A 路用 RETFound-CFP 1024 维特征。")
    B("渗出区域加权：w=1+λ·mask（λ=8），只检测渗出（亮黄阈值 + 连通块过滤）；出血检测未实现，"
      "血管仅用于从渗出 mask 中减除假阳性，不是“血管降权”。")

    # ============================================================
    H("3. 失效机理链", 1)

    H("3.1 总览", 2)
    TABLE(
        ["环节", "一句话机理", "关键证据"],
        [
            ["VAE 模糊天花板", "逐像素平均的数学性质，加容量/轮数不可逃逸", "FID 188.4 但 C2ST 0.9992；800 vs 1200 轮肉眼无差"],
            ["GAN 判别器记忆", "小样本下判别器必胜，生成器收不到有效梯度", "DCGAN FID 229.8；出血代理 0.000；C2ST 1.0"],
            ["MSE 病灶溶解", "暗背景占 95% 像素时，MSE 最优解是“安全的暗色”", "MSE vs L1+LPIPS 受控消融；Palette 复现"],
            ["L1+LPIPS 修复", "对离群值不敏感 + 感知约束，显著缓解溶解", "出血代理 0.254→1.137（4.5×）；FID 200.2→178.8"],
            ["数据量与迁移", "loss 收敛 ≠ 生成成立；迁移胜出但受信息量限制", "D3 vs D4 四组裁决；单变量教训"],
            ["评估盲区", "通用分布指标对医学语义差异近乎失明", "FID 接近但 C2ST/病灶代理差异巨大"],
        ],
        widths=[1.5, 3.0, 2.5],
    )

    H("3.2 VAE：模糊天花板", 2)
    P("VAE 的失败不是训练不足：loss 从 1639 降到 290–380，容量翻倍与 1200 轮训练都没有改变模糊。"
      "FID 188.4 在通用指标下并不难看，但 C2ST AUC 0.9992 说明分类器几乎总能识破；"
      "出血代理指标 0.700 主要来自整体亮度而非病灶。人工分 10，六维门控分 26.6。")

    H("3.3 GAN：判别器记忆", 2)
    P("DCGAN 训练 1500 轮，FID 229.8；判别器过拟合后生成器梯度消失，输出退化为低质量解。"
      "出血代理 0.000、C2ST 1.0、人工分 20、门控分 12.9。WGAN-GP 标准版两轮内 Critic 爆炸；"
      "削弱版 loss 稳定但生成图 2770 轮不变；StyleGAN2-ADA 因环境不兼容未运行。")
    FIGURE("fig_dcgan", "图 3-1  DCGAN 训练 1500 轮后的输出：判别器过拟合，生成器无法继续学习")

    H("3.4 MSE 病灶溶解：受控消融", 2)
    P("同一 FiLM 条件 UNet，只换损失：MSE 臂 vs L1+LPIPS 臂。这是本项目最干净的因果证据。")
    TABLE(
        ["损失臂", "出血代理指标", "FID", "C2ST AUC"],
        [["MSE（DL-EX-013）", "0.254", "200.2", "0.989"],
         ["L1+LPIPS（DL-EX-014）", "1.137（4.5×）", "178.8", "0.915"]],
        widths=[1.8, 1.6, 1.2, 1.2],
    )
    P("注意：出血代理指标是颜色阈值计数（见 5.2），只能作相对比较；FID 仅差约 11%，"
      "代理指标却差 4.5 倍，说明通用分布指标对这一医学语义差异近乎失明。")
    FIGURE("fig_cond800", "图 3-2  条件扩散 epoch 800：亮病灶随训练被拉暗（MSE 溶解）")

    H("3.5 L1+LPIPS 修复与双尺度评分", 2)
    P("L1+LPIPS 臂训练 780 轮，直到最后一个检查点（775）白斑、出血点仍清晰可辨，微细血管保持"
      "对比度与连续性；人工分 85–90，六维门控分 45.2（v1，组内相对归一化）。"
      "但干净样本率约 25–30%，展示图必须手工挑选并注明。")
    FIGURE("fig_film", "图 3-3  FiLM DDPM + L1+LPIPS epoch 700：细节最丰富的最佳档")

    H("3.6 数据量假说与 D3/D4 裁决", 2)
    P("D1（76 张 @256，bd64）0/8 可用，但同时改了容量与分辨率，不能归咎数据量单变量；"
      "D3（A 域最佳权重微调 193 张 @128）与 D4（同数据从零 @128，1100 轮）做四组并排裁决："
      "两个权重臂都判微调胜出。D3 细节更完整但看得出生成；D4 的“拍摄感”更好但噪声多、细节乱。"
      "两者共同瓶颈是数据信息量，而不是容量。")
    TABLE(
        ["模型", "初始化", "数据/轮数", "裁决"],
        [["D3", "A 域 FiLM L1+LPIPS 权重", "193 张 @128；1100 轮", "EMA 臂细节优于 D4；部分样本近替换级"],
         ["D4", "从零", "193 张 @128；1100 轮", "风格/拍摄感学到，但噪声多、细节远逊 D3"]],
        widths=[0.8, 2.0, 1.8, 2.4],
    )

    # ============================================================
    H("4. CFM 主线结果", 1)

    H("4.1 CFM@128 vs D3", 2)
    P("同初始化（D3 的 FiLM UNet）、同数据（193 张）、同轮数、同种子，唯一差异是目标函数与路径。")
    TABLE(
        ["指标", "D3", "CFM@128"],
        [["FID ↓", "222.7", "178.9"],
         ["KID ↓（无偏）", "0.118±0.010", "0.158±0.007"],
         ["C2ST AUC ↓（理想 0.5）", "0.859", "0.940"],
         ["疑似出血占比 ↑（代理）", "0.754", "0.320"],
         ["BRISQUE ↓", "42.8", "38.0"]],
        widths=[2.2, 1.8, 1.8],
    )
    P("FID 明显改善，但 KID 反而更差、C2ST 更高——说明 CFM@128 在“分布距离”上占优，"
      "在“判别难度”和“病灶代理”上不如 D3。这是真实性与病理保真的取舍，不能只挑一项报。")

    H("4.2 CFM@256：原生高分辨率", 2)
    P("在 combined_sq256（193 张）上以 bd128 训练 1420 轮，原生 256 生成。")
    TABLE(
        ["指标", "D3", "CFM@256"],
        [["FID ↓", "222.69", "121.84（−45%）"],
         ["KID ↓（无偏）", "0.118±0.010", "0.084±0.008"],
         ["BRISQUE ↓（真实 11.02）", "42.83", "13.21"],
         ["颜色均值距离 ↓", "0.2996", "0.0345"],
         ["C2ST AUC ↓", "0.859", "0.906"],
         ["疑似出血占比 ↑（代理）", "0.754", "0.204"],
         ["MS-SSIM（越低=多样性越好）", "0.1596", "0.4522"]],
        widths=[2.4, 1.8, 2.0],
    )
    P("作者目检 97/100（“差点没看出来”），这是首次通过作者自检线——不是医生盲评。"
      "口径限制：FID/KID 在 72 real + 60 fake 下无方差估计（KID 现有 std，但只是子集方差，不是训练种子方差）；"
      "参照集与训练池同源；C2ST 0.906 距 0.5 仍远；D3 的低 MS-SSIM 可能同时包含多样性与混乱，不能简单当作优势。")
    FIGURE("fig_cfm_curve", "图 4-1  两阶段 CFM 训练曲线（细线原始、粗线滑动平均）")
    FIGURE("fig_same_slot", "图 4-2  同槽对比：真实 / D3_ema / D3+SDEdit+SR / CFM@128+SR / CFM@256 原生")
    FIGURE("fig_bdomain", "图 4-3  B 域三模型关键指标（越低越好）")
    FIGURE("fig_sr", "图 4-4  CFM 的自精修与超分链路（6/7 行对照）")

    H("4.3 渗出区域加权（旧实验，仅探索性）", 2)
    P("动机：CFM@256 的病灶代理指标明显低于 D3。做法：用颜色阈值生成渗出权重图（亮黄 + 连通块过滤，"
      "覆盖率>15% 清零），对速度预测误差施加 w=1+λ·mask 的加权 L1（λ=8），从 CFM@256 final 续训 150 轮。"
      "仅渗出被加权；出血检测未实现。")
    P("旧口径结果：渗出代理 +36%、出血代理 +24%（附带，不作归因）、FID 121.84→115.22、"
      "BRISQUE 13.21→14.49。")
    P("2026-11-06 复核发现两个训练缺陷，使这批数字不再构成方法证据："
      "①加权项分母只计单通道，损失恒为预期加权均值的 3 倍（训练日志实证 3.057×）；"
      "②图像随机翻转/旋转时权重图与血管 mask 未跟随，权重实际施加在错位区域。"
      "因此本节只作探索性记录；修正后的两臂（λ=0 对照与 λ=8）待从零重训。")
    FIGURE("fig_lesion", "图 4-5  同种子对比：真实 severe / CFM@256 / 渗出加权版（λ=8，旧实验）")

    H("4.4 TTA 组件与检索条件消融（2026-10-26）", 2)
    P("统一协议：128px、60 张、采样 seed=100（8 张/批连续流）、Euler 50 NFE、B 域指标在 256 口径算。")
    TABLE(
        ["方法", "FID ↓", "KID ↓（无偏）", "疑似出血占比 ↑*", "疑似渗出占比 ↑*", "C2ST ↓"],
        [
            ["cfm128 基线", "178.9", "0.158±0.007", "0.320", "0.445", "0.940"],
            ["+时间对齐 align", "151.8", "0.121±0.008", "0.375", "0.571", "0.943"],
            ["+文本条件 text", "148.2", "0.113±0.007", "0.420", "0.575", "0.916"],
            ["去时间条件 notime", "183.7", "0.168±0.008", "0.522", "0.252", "0.928"],
            ["检索 featC（手工）", "164.9", "0.136±0.010", "0.567", "—", "0.836"],
            ["检索 featA（RETFound）", "162.8", "0.137±0.010", "0.463", "0.691", "0.930"],
        ],
        widths=[1.9, 0.9, 1.3, 1.4, 1.4, 1.0],
    )
    P("* 颜色阈值疑似像素占比（代理指标，未验证）。featC 渗出 1.029 为极小分母伪影，已从核心表移除。"
      "另：全部条件模型均受增强错位缺陷影响，FID/KID 排名只作相对参考。")
    B("时间语义对齐有效：FID 178.9→151.8，证实 DDPM→CFM 的时间反转是隐性损失源。")
    B("文本条件是最佳单条件：FID 148.2、KID 最低。")
    B("去时间条件在眼底不成立：FID 183.7，比基线更差——与 TTA-Flow 在 OCT 上的结论相反。")
    B("检索条件不改善 FID（164.9/162.8），但 C2ST 更低、多样性 +7.8%；走“病理保真 + 多样性”路线。")
    FIGURE("fig_ablation", "图 4-6  128px 受控消融：同一采样协议下的六种条件方案（同列同序号，非配对）", width=6.2)

    H("4.5 两个负结果", 2)
    P("①轨迹对齐直方图匹配（TTA-Flow 思路）：真实 severe 图经参考轨迹 CDF 匹配 + ODE 重建后"
      "过曝/高饱和，不可用。OCT 灰度单通道与眼底 RGB 三通道的色彩动态范围不同，逐通道匹配破坏色彩耦合。")
    P("②Tiger-Model 零样本：罕见病文本引导扩散（OCT/CFP 混合预训练）输出灰度类超声环，"
      "完全不像眼底。跨模态预训练先验不能迁移到彩照眼底。")

    # ============================================================
    H("5. 评价体系复核", 1)

    H("5.1 两层指标与六维门控", 2)
    P("通用层指标衡量分布与感知，医学层指标衡量结构；六维门控把“低层代理可被博弈”的维度乘上"
      "“整体像真实”的门，避免模型只在单一代理上刷分。需要强调：任何单个数字都不能证明生成图"
      "在医学上正确，指标必须分工回答不同问题。")

    H("5.2 病灶代理指标的失效反例", 2)
    P("当前“出血/渗出保留率”是颜色阈值计数：出血 R<0.42 且 R−G>0.04 且 R−B>0.04；"
      "渗出 min(RGB)>0.42 且 mean>0.48。未对照像素级标注验证。反例：DCGAN 的全白图渗出存在率=1.0、"
      "保留率=40×；VAE 暗糊图出血存在率=0.91。该指标实际主要跟随整体亮度。"
      "真实渗出中位仅 0.067% 像素，阈值被压到 0.034%，任何带视盘的图都能过；保留率>1 是除以极小"
      "分母的伪影。因此本报告统一改称“疑似出血/渗出像素占比（代理，未验证）”，只作相对比较。")

    H("5.3 KID：旧实现的缺陷与无偏重算", 2)
    P("旧实现声明了子集自助，但实际只算了一次含核矩阵对角的全矩阵估计，且没有方差输出——"
      "本质是有偏点估计。2026-11-06 改为无偏子集自助（排除对角，100 个子集，返回 mean±std），"
      "并对全部已有模型重算。旧值保留为 kid_old，新旧不可混比。")
    TABLE(
        ["A 域模型", "KID_old（有偏）", "KID_new（无偏）", "std"],
        [["VAE Large", "0.189", "0.186", "0.008"],
         ["DCGAN", "0.204", "0.202", "0.006"],
         ["基础 DDPM", "0.139", "0.138", "0.008"],
         ["条件扩散", "0.132", "0.131", "0.009"],
         ["FiLM MSE", "0.126", "0.124", "0.009"],
         ["FiLM+L1+LPIPS", "0.107", "0.104", "0.008"]],
        widths=[1.8, 1.6, 1.6, 0.8],
    )
    TABLE(
        ["B 域模型", "KID_old（有偏）", "KID_new（无偏）", "std"],
        [["cfm128", "0.163", "0.158", "0.007"],
         ["cfm256", "0.090", "0.084", "0.008"],
         ["cfm256_lesion（旧实验）", "0.076", "0.069", "0.007"],
         ["cfm128_align", "0.127", "0.121", "0.008"],
         ["cfm128_text", "0.118", "0.113", "0.007"],
         ["cfm128_notime", "0.174", "0.168", "0.008"],
         ["cfm128_featC", "0.143", "0.136", "0.010"],
         ["cfm128_featA", "0.142", "0.137", "0.010"],
         ["D3", "0.129", "0.118", "0.010"]],
        widths=[2.0, 1.5, 1.5, 0.8],
    )

    H("5.4 C2ST 的现实检查", 2)
    P("C2ST 训练一个真假二分类器，AUC 越接近 0.5 越难区分。本报告所有方法在 0.836–0.944，"
      "距 0.5 仍有 0.34–0.44——全部可被轻易判别。方法间差异常小于 C2ST 的折间标准差，"
      "因此只能作相对排序，不能声称“已不可判别”。")

    H("5.5 多样性与记忆检测", 2)
    P("LPIPS 成对多样性：cfm128 基线 0.321，featC/featA 0.346（+7.8%）。"
      "最近邻 SSIM 记忆检测：所有学习模型复制率 0（SSIM>0.85 占比为 0）。"
      "注意：多样性高也可能来自伪影，必须与真实性和结构合理性一起看。")

    # ============================================================
    H("6. 研究自查与更正（2026-11-06）", 1)
    P("本节只给结论与指针，完整证据见 EXPERIMENTS 执行记录 21 与 advisor_review_20261106/。")

    H("6.1 两个训练缺陷（已修复）", 2)
    P("①加权损失 3× 尺度：权重形状 [B,1,H,W]，速度预测 [B,3,H,W]；分子按 3 通道累加、分母只计 1 份，"
      "加权项恒为预期加权均值的 3 倍（λ=0 时也不等于普通 L1）。训练日志实证 0.3312/0.1083=3.057×。"
      "修复为 w=(1+λ·lw).expand_as(predicted)，并尊重 mse/l1；单测 7/7 通过。")
    P("②增强错位：图像随机水平翻转（p=0.5）+15° 旋转，但权重图与血管 mask 未跟随；"
      "cond_transform 构造后从未传入数据集。诊断：翻转后权重与渗出区域重叠从 0.551 降到 0.000。"
      "修复为一次采样 flip/angle，同一组参数作用于图像、条件与权重图；默认路径行为不变。"
      "该缺陷波及所有条件模型（D3、CFM@128/256、10-26 全部消融），其“条件跟随”类结论需加限制；"
      "图像质量类指标仍可参考。")
    FIGURE("fig_misalign", "图 6-1  增强错位诊断：图像翻转后权重图未跟随，权重覆盖区域与渗出区域完全错开")

    H("6.2 数据与指标更正", 2)
    B("HR/DR：全仓库约 24 处误写已更正；术语表加入“本项目一律写 HR”的红线。")
    B("报告 MS-SSIM 方向：第 6 章两处方向标注写反（误把多样性指标当作重建质量），已改为“越低越好（多样性）”。")
    B("KID：无偏重算并输出 std；A 域 6 个模型、B 域 9 个模型已回填。")
    B("FID 口径：明确为“训练分布拟合”，禁止跨域比较。")
    B("病灶指标：降级为颜色阈值代理；featC 渗出 1.029 移出核心表。")
    B("跨 split 审计：32 对（5 exact + 27 near），3/3 已知像素级重复命中；建议按眼/序列重划分。")
    B("交付批次：旧进程因“整批落盘”中断丢图为 0；generate.py 已加 --resumable（逐批落盘、断点续跑）。")

    H("6.3 影响矩阵", 2)
    TABLE(
        ["产物", "训练缺陷影响", "当前处理"],
        [["cfm256_lesion（渗出加权）", "3× 尺度 + 增强错位", "结果作废为方法证据，仅探索性记录；待从零重训两臂"],
         ["cfm256 / cfm128 / D3 / 10-26 消融", "增强错位（条件）", "图像质量类指标保留并加限制；条件跟随类结论降级"],
         ["VAE / GAN / FiLM / 传统 ML", "无", "保留"],
         ["FID/KID/病灶代理", "指标实现问题", "KID 已重算；FID 加口径；病灶指标降级"],
         ["下游 QWK 0.747", "数据集跨 split 泄漏", "保留为参考值，注明未知乐观偏差；建议重划分"]],
        widths=[2.6, 1.8, 3.2],
    )

    # ============================================================
    H("7. 下游验证与交付（Q4）", 1)

    H("7.1 交付管线", 2)
    P("交付目标：生成 severe 新图 → 作者筛选 76 张 → 老师审核 → 替换 train 中的翻转副本 → 重训分类器。"
      "generate.py 新增 --resumable：逐批落盘、跳过已完成批次、每批独立播种；"
      "单测验证整批跳过与单张补生成。旧管线的 delivery_final 已产出 200 张，仅用于初筛；"
      "正式交付待修正管线重生成。cfm256_lesion 旧模型不再作为交付候选。")

    H("7.2 TSTR 现状", 2)
    P("既有 TSTR 使用 B1（Stable Diffusion + LoRA）线，不是 CFM；test 全真实（n=71），3 个分类器种子。")
    TABLE(
        ["注入臂", "QWK", "severe 召回", "bAcc", "macroF1"],
        [["纯真实基线（n=552）", "0.7365±0.025", "0.8205", "0.6579", "0.6580"],
         ["旧 A 域生成器 +87（n=639）", "0.7324±0.037", "0.8718", "0.6471", "0.6222"],
         ["B1 原始 +60（n=612）", "0.7352±0.013", "0.7949", "0.6950", "0.6982"],
         ["B1 精修 0.25 +60（n=612）", "0.7354±0.037", "0.8205", "0.7047", "0.7133"]],
        widths=[2.4, 1.4, 1.1, 1.0, 1.0],
    )
    P("解读：QWK 四臂全部在噪声内持平（被三个非 severe 类主导）；B1 两臂首次把 bAcc/macroF1 推正。"
      "但 severe 召回噪声极大（test severe 仅 13 张，1 张=7.7pp），且这是 B1 线而非 CFM，"
      "只能作为“合成数据可用于下游”的数据点，不是方法裁决。")

    H("7.3 A/B 实验设计（待执行）", 2)
    B("实验 A（替换）：真实 severe 与生成 severe 按 0/25/50/75% 替换，总样本数固定。")
    B("实验 B（补充）：保留全部真实 severe，额外加入 10/20/30 张生成图。")
    B("5 路对比：基线 / 传统增强 / DDPM 生成 / CFM 生成 / 渗出加权 CFM，同一真实 test。")
    B("数据隔离：生成器只见 train；val/test 全真实；尽量按眼/序列划分。")
    B("医生盲评：整体真实性、解剖结构、病灶真实性、伪影、可接受性五维；真实图混入、随机编号。")

    # ============================================================
    H("8. 结论与边界", 1)

    H("8.1 当前数据/配置下可支持的结论", 2)
    B("三代模型的失效机理（模糊、判别器记忆、MSE 溶解）在本数据上可重复；L1+LPIPS 能显著缓解溶解。")
    B("相同初始化/数据/轮数下，CFM 的分布与感知指标优于 DDPM 微调线；CFM@256 达到作者自检 97/100。")
    B("时间语义对齐与文本条件带来相对改善；去时间条件在眼底不成立。")
    B("检索条件不改善 FID，但提升多样性与判别难度。")

    H("8.2 尚不能支持的结论", 2)
    B("不能声称 CFM 在医学结构保真上全面优于其他方法——病灶代理指标仍低于 D3。")
    B("不能声称渗出加权有效——旧实验受两个缺陷影响，修正实验未完成。")
    B("不能声称合成数据提升下游分级——CFM 版 TSTR 未做，既有 B1 结果 QWK 持平。")
    B("不能声称留出泛化——FID 参照与训练集同源，数据集本身存在跨 split 重复。")
    B("不能声称方法优越性的因果结论——单训练种子、代理病灶指标、无医生盲评。")

    H("8.3 局限", 2)
    B("训练数据仅 193 张；severe 唯一底图 76 张；test severe 仅 13 张。")
    B("FID 参照与训练池同源；KID 方差大；C2ST 全部仍可判别。")
    B("病灶指标是颜色阈值代理，未对照像素级标注。")
    B("所有历史训练为单 seed 42；条件模型受增强错位影响；8GB 显存限制容量与分辨率。")
    B("医生盲评尚未执行；作者自检不等于临床验证。")

    # ============================================================
    H("参考文献", 1)
    refs = [
        "[1] Kingma & Welling (2013). Auto-Encoding Variational Bayes.",
        "[2] Goodfellow et al. (2014). Generative Adversarial Nets.",
        "[3] Radford et al. (2015). Unsupervised Representation Learning with DCGANs.",
        "[4] Gulrajani et al. (2017). Improved Training of Wasserstein GANs.",
        "[5] Karras et al. (2020). Training GANs with Limited Data (StyleGAN2-ADA).",
        "[6] Ho et al. (2020). Denoising Diffusion Probabilistic Models.",
        "[7] Song et al. (2020). Denoising Diffusion Implicit Models.",
        "[8] Saharia et al. (2022). Palette: Image-to-Image Diffusion Models.",
        "[9] Lipman et al. (2022). Flow Matching for Generative Modeling.",
        "[10] Zhou et al. (2023). A foundation model for generalizable disease detection from retinal images. Nature.",
        "[11] Hucke et al. (2025). Test-Time Adaptation in OCT Using Trajectory-Aligned Time-Independent Flow.",
        "[12] FSFM (2025). Few-Shot Flow Matching for retinal image generation (project-internal reference).",
    ]
    for r in refs:
        doc.add_paragraph(r)

    # ============================================================
    H("附录 A 实验记录（EX-001～EX-018）", 1)
    P("以下记录从 2026-10-24 版报告原样迁移，保留训练过程、问题与修复细节，便于追溯；"
      "其中涉及“病灶加权”的表述按 2026-11-06 复核结论理解——实际为渗出加权，且旧结果仅探索性。"
      "EX-017 标题中的“进行中”是当时状态：该消融已于 2026-10-26 完成，结果见正文 4.4；"
      "附录保留原记录以如实反映时间线。")
    sectPr = body.find(qn("w:sectPr"))
    for el in appendix:
        if sectPr is not None:
            sectPr.addprevious(el)
        else:
            body.append(el)

    # ============================================================
    H("附录 B 2026-11-06 更正清单", 1)
    TABLE(
        ["项目", "问题", "处理"],
        [["HR/DR 术语", "约 24 处把本项目疾病写成 DR", "改为 HR；docs/14 加术语红线"],
         ["MS-SSIM 方向", "报告第 6 章两处写成“越接近 1 越好”", "改为“越低越好（多样性）”"],
         ["KID 实现", "子集自助为死代码；含对角有偏；无方差", "无偏子集自助 + std；15 个模型重算"],
         ["FID 口径", "参照集与训练池同源，易被误读为泛化", "统一声明为训练分布拟合；禁止跨域比较"],
         ["病灶指标", "颜色阈值计数被当作病灶保真", "降级为代理指标；featC 1.029 移出核心表"],
         ["渗出加权实验", "3× 损失尺度 + 增强错位", "结果作废为方法证据；修正两臂待重训"],
         ["条件模型", "图像增强时 cond/权重未同步", "代码修复；历史模型结论加限制"],
         ["跨 split 泄漏", "官方 split 含同图/近重复", "审计 32 对；建议按眼/序列重划分"],
         ["交付批次", "整批落盘，中断即丢图", "generate.py --resumable；旧 200 张仅初筛"]],
        widths=[1.6, 2.8, 2.6],
    )

    # 保存
    doc.save(str(NEW))
    print(f"[done] {NEW}")


if __name__ == "__main__":
    main()
