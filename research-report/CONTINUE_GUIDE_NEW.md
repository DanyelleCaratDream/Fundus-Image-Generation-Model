# Continue Guide - 2026-08-02（v15 — Phase A ✅ + Phase B ✅ + 第五节收尾 ✅ + 可视化图表 ✅ + 综合评分方案 ✅ + 目录重组 ✅）

> **当前状态：Phase A ✅ + Phase B ✅ + 第五节收尾 ✅ + 综合评分方案 ✅（2026-08-02）→ ⚠️ 目录重组 ✅（Fundus-* 已搬入 `generate_project/deep_learning/`，新建 pretrained/machine_learning/transfer_learning + classify/ 占位，所有文档路径已更新、代码零改动）→ REPORT（原版）.docx 待用户改排版 → 老师已答复：① 只做 DR 分级分类（不框出/检测）；② 330 张需提前标注等级（老师/医生标）；③ 分级标准 = KW（Keith-Wagener）I-IV 级高血压视网膜病变（老师说的 HR 即此，用户 2026-08-02 确认）→ 等 330 标注到位 + KW 细节定 Phase C/D**
> **⚠️ 目录重组影响所有路径**：凡文档出现 `Fundus-Diffusion/GAN/VAE` 均指 `generate_project/deep_learning/` 下；生成命令的 `../../eval_data` → `../../../eval_data`（深一层）。旧位置已不存在。
> **⚠️ 注意：第五节还有少数"待改/待完善"（颜色表措辞、evaluation_report.docx 与 md 不同步），以及 REPORT.docx 收尾**
> **项目方向（v6 转向）：生成"相似但不相同"的重度眼底图，扩充 DR 分级分类器训练集**
> **模型最佳方案：FiLM DDPM + L1 + LPIPS 780 轮，85~90/100 分**
> **双远端：** GitHub + Gitee

---

## 一、项目方向与老师反馈（2026-08-01 确认）

**新目标：** 利用 330 张严重症状眼底彩照，生成「相似但不跟原图一样」的图片，扩充 DR 分级分类器训练集（可能也做病灶类型判断）。现实里重度眼底图稀缺 → 分类器对重度类偏颇 → 需合成重度图补足。

**老师两个 Bug 反馈：**
1. **Bug 1 评分标准必须全**：通用指标（FID/召回率/IS/KID 等）+ 自设计指标全部要算，先用历史结果重算，再调研更适合眼底图的
2. **Bug 2 方法不限于深度学习**：OpenCV+ML、预训练、迁移学习（Stable Diffusion）都可以试

**数据现实：** 330 张全重度（KW IV 级）、无标签、无病灶 mask；有 330 张血管骨架 mask（`generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/`）。下游分类器验证需有标签数据 —— 分级标准 = **KW（Keith-Wagener）高血压视网膜病变 I-IV 级**（老师此前说的"HR"即此，用户 2026-08-02 确认，IV 级最重含视乳头水肿）；330 张由老师/医生标注等级；其他程度数据在别的同学手里（小组汇总由老师协调，不担心分布）。

### 老师最新沟通（2026-08-01 下午）
- **老师要求**：总结昨天会议（讨论内容 / 存在问题 / 下一步安排）；并重申方向——①寻找好的评分标准；②拓展思路找非从零训练的生成方法（预训练模型 / 传统 ML+CV / 迁移学习）；③让同学们先看后面的分类模型内容，为分类训练打基础
- **已回复**：起草并发了微信会议总结（约 600 字），内容含：讨论内容（模型对比结论）、存在问题（指标偏通用 / 30~40% 生成图有伪影）、下一步（评分标准 → 已备好简单评估文档/表格发同学参考 + 拓展方法 + 分类基础）、以及两个提问（见下）
- **发问 → 老师已答复（2026-08-02）**：
  1. **只做 DR 分级分类**（不做病灶框出/检测 → 无需 YOLO，纯分类网络即可）
  2. **330 张需提前标注等级**（由老师/医生标注；330 张全重度，其他程度在别的同学手里，小组汇总由老师协调）
  3. **分级用 HR 标准**（非 APTOS/ICDR 的 0-4）→ **2026-08-02 用户确认 = KW（Keith-Wagener）高血压视网膜病变 I-IV 级**，本项目生成 IV 级（最重）；病灶重点 = 火焰状出血/硬性渗出/棉絮斑/血管改变
- **迁移学习说明（用户曾问）**：没有被排掉。报告 4.8 分析过"医学图像生成从零训练是常态、预训练帮助有限"，但老师 Bug 2 已重新放开；Phase C3（SD LoRA）在指引里列为可选探索
- **可发同学的文件已就绪**：`research-report/evaluation_report.docx`（2026-08-02 已把原 evaluation_metrics.csv/.xlsx 数据并入，三份合成一份）

---

## 二、已完成（截至 2026-08-01 晚）

| 项 | 状态 |
|------|:--:|
| 工程前置文档（docs/ 9 份） | ✅ |
| 答辩 PPTX（29 页）+ interview-prep（Q1-Q14）+ REPORT.docx | ✅ 用户精修 |
| 最佳模型 FiLM+L1+LPIPS 训练 | ✅ 85~90/100 |
| **Phase A 两个障碍解决**（weights_only + 旧版 UNet 兼容） | ✅ |
| **Phase A 评估图生成**（6 模型 × 300 张 → eval_data/） | ✅ |
| **Phase A 评估脚本** eval/metrics_common.py（9 项 + 颜色统计） | ✅ |
| **Phase A 6 模型评估完成**（JSON + CSV + evaluation_report.md） | ✅ |
| **Phase A 专用层自设计指标** eval/metrics_fundus.py（病灶/血管/记忆检测/C2ST/BRISQUE/Vessel Dice） | ✅ 2026-08-01 晚补齐 |
| 报告/表格转档 + 文件整合（evaluation_report.docx；原 evaluation_metrics.csv/.xlsx + metrics_research.md 并入合并版） | ✅ 2026-08-02 |
| **Phase B 评分标准调研**（结论并入 evaluation_report.md 第二部分） | ✅ 2026-08-02 |
| **第五节收尾**（2026-08-02）：eval_data .gitignore、命令沉淀 Work-Guide、eval/README.md、排名#3措辞修正、50/100步DDIM实测、VAE颜色假象确认、评估图全量扫描 | ✅ 详见第五节 |
| **数据提纯工具**（2026-08-02）：fundus/rotate_augment_check.py（旋转45/90/135/180合成2×2对比图）+ 使用说明.txt，同学筛选旋转安全图 | ✅ |
| **评估可视化图表**（2026-08-02）：eval/plot_metrics.py（可复用，新增方法重跑即入图）→ research-report/figures/ 四图，已整合进 evaluation_report.md | ✅ 详见第五节 |
| **REPORT（原版）.docx 插入评估章节**（新「2.评估指标与评分标准」+ 原章节重编号 +1 → REPORT（原版）_with_metrics.docx，原文件未动） | 🟡 用户 2026-08-02 将自行改排版，待替换 |
| **综合评分方案：需求确认 + 文献调研 + 公式设计 + 实现（score_scheme.py 六维门控 0-100 总分 + 校准表 + scorecard 图，见第五节 5）** | ✅ 2026-08-02 |

---

## 三、Phase A 评估结果（已全部跑完）

### 总表（真实 330 张 vs 每模型 300 张，DDIM 50 步，seed 42）

| 模型 | FID↓ | KID↓ | MMD↓ | IS↑ | Precision↑ | Recall↑ | Density↑ | Coverage↑ | 1-NN(0.5) | MS-SSIM↓ | LPIPS↓ | 人工分 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **FiLM+L1+LPIPS** | **178.8** | **0.107** | **0.001** | **2.46** | 0.030 | 0.442 | 0.008 | 0.030 | **0.984** | 0.201 | 0.663 | 85-90 |
| FiLM MSE | 200.2 | 0.126 | 0.006 | 2.26 | 0.007 | **0.742** | 0.001 | 0.006 | 0.975 | **0.187** | 0.791 | 85 |
| 条件扩散 | 187.2 | 0.133 | 0.003 | 2.34 | 0.033 | 0.567 | 0.009 | 0.030 | 0.983 | 0.251 | 0.721 | 70 |
| 基础 DDPM | 188.2 | 0.139 | 0.003 | 2.14 | **0.053** | 0.473 | **0.013** | **0.036** | 0.979 | 0.270 | 0.710 | 75 |
| DCGAN | 229.8 | 0.204 | 0.007 | 1.18 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.652 | 0.726 | 20 |
| VAE Large | 188.4 | 0.189 | 0.004 | 2.30 | 0.013 | 0.000 | 0.003 | 0.009 | 1.000 | 0.361 | **0.455** | 10 |

### 核心结论
1. **评估体系验证有效**：film_l1lpips 分布指标最优、DCGAN 全面最差，与人工评分方向一致。
2. **扩散方法集体胜出**：4 个扩散模型 Recall 0.44-0.74 vs GAN/VAE 全 0 → 验证转向扩散决策正确。
3. **损失权衡量化**：FiLM MSE 更"自由发散"（Recall 0.742 最高、MS-SSIM 最低）、L1+LPIPS 更"收敛保真"（FID/KID/MMD 最优）。
4. **系统色偏发现**：所有扩散模型 B 通道偏正（真实 B=-0.71）→ color_correct 有改进空间。

### 专用层补充（2026-08-01 晚，自设计指标，回应老师 Bug 1 的"自设计"）
- **L1+LPIPS 显著抗病灶溶解**：出血保留率 1.137（病灶全保留）vs MSE 版 0.254（丢 76%）
- **C2ST**：film_l1lpips AUC=0.915 最低（最难被识破）→ 与 FID/1-NN 排序自洽
- **"相似但不相同"强验证**：所有生成模型复制率 0%、最近邻 SSIM 0.07~0.33（真实-真实自身 0.548/16.4%）→ 生成图不复制训练图
- 详细表 + 解读见 evaluation_report.md 第「二·补」节

---

## 四、Phase A 已解决的技术点（复现/扩展用）

1. **DCGAN/VAE `weights_only`**：generate.py 的 `torch.load` 已加 `weights_only=False`（DCGAN:64、VAE:68）。VAE large 还需 `--dim 64 --latent_dim 256`。
2. **cond/base_cj 旧版 UNet 兼容**：generate.py 已内置自动检测（`block2.0.` 键）→ 禁用 FiLM + key 重映射（`block2.0→norm2`、`block2.3→conv2`），`base_dim=64`。验证 missing=0/unexp=0。
3. **Inception/LPIPS/torchvision 权重**：已下载到 `~/.cache/torch/hub/checkpoints/`（需 VPN 从 GitHub 下载 FID 权重）。`pytorch-fid` 无 IS（1008 维），IS 用 torchvision InceptionV3。
4. **MS-SSIM 需 ≥160px**：128px 输入内部 resize 256 再算。
5. **prdc 会污染 stdout**：json 模式下已用 redirect_stdout 抑制。

---

## 五、⚠️ 待改 / 待完善（compact 后第一步，按用户指示）

> 用户明确提示："任务还没完全结束，还有一些内容需要改改的"。

### 1. 评估报告待完善
- [x] **综合排名第 3 名措辞**（2026-08-02 已修正）：evaluation_report.md 5.4 改为"基础 DDPM / 条件扩散 并列接近——分布/保真指标 cond 略优（FID 187 vs 188、KID/IS/Recall 更优），人工评分 base_cj 略优（75 vs 70），按人工列 base_cj 前"
- [x] **base_cj 画像核对**（2026-08-02）：数据属实（P/D/C 确实全场最高 0.053/0.013/0.036、Recall 0.473 低于 cond/film），无需改
- [x] **可视化图表**（2026-08-02）：`eval/plot_metrics.py`（自动扫描 eval_data/ JSON，可复用，新增方法重跑即入图）生成四图 → `research-report/figures/`（general/color/fundus/radar），已整合进 evaluation_report.md（总表后/颜色后/专用层后 + 5.5 雷达图小节）
- [ ] 确认颜色统计表的解读措辞是否合适
- [x] **evaluation_report.docx 与 md 同步**（2026-08-02）：pandoc 3.8 直接重转（无需样式模板），含 5.6/5.7/5.8 全部新章节 + 5 张图（9 表格 160 段落已验证）；设计文档 docs/09 同步转 09-Score-Scheme-Design.docx

### 2. 潜在遗留问题（compact 前未能验证）
- [x] **50 vs 100 步 DDIM 对比**（2026-08-02 已实测）：同 seed 生成 8 张×2 模型×2 步数 → base_cj SSIM 0.957 / cond 0.977 → **50 步已收敛**，Phase A 评估有效；对比图在 `eval_data/_step_compare/`（gitignore 内）
- [x] **VAE 颜色"假象"确认**（2026-08-02）：解释有数据支撑（颜色三指标全场最优 0.024/0.040/0.0018 + Recall=0 + 记忆 NN-SSIM 0.329 最高"平均脸"），准确
- [x] **评估图肉眼异常抽查**（2026-08-02 全量扫描 1800+ 张）：仅 DCGAN 21 张近空白（std<3，符合其退化画像），其余模型 0 异常、亮度正常

### 3. 工程整理
- [x] **eval_data .gitignore**（2026-08-02）：已加 `eval_data/*/`，git 只跟踪根目录 JSON/CSV（验证 check-ignore 生效）
- [x] **命令沉淀**（2026-08-02）：生成（三/节，DDIM 50 步）+ 评估（四节 metrics_common/metrics_fundus）已写入 docs/08-Work-Guide.md
- [x] **metrics_common.py 放置**（2026-08-02）：保持 eval/ 不动（多处文档引用，挪动纯churn），新增 `eval/README.md` 说明两脚本用法

### 3.5 REPORT.docx 评估章节增强（2026-08-02，待用户收尾）
- [ ] **用户自己改 REPORT（原版）_with_metrics.docx 的排版**（插入的新章节 2.1~2.10 + 4 张评估表，用户发现排版问题自行修改）
- [ ] 用户改好 → 确认是否替换原 `REPORT（原版）.docx`（原文件至今未动）
- [ ] 替换后删除临时脚本 `research-report/_insert_metrics.py`

### 5. ⭐ 综合评分方案（✅ 已完成 2026-08-02，实现见 `eval/score_scheme.py` + 报告 5.6/5.7）

**背景**：两层 ~30 指标全测一遍就完，没有综合评分；5.4「加权各维度」只是标题无公式；人工分是单人不规范肉眼估分。已实现：用人工分做"金标准"校准指标（哪些合适/哪些有偏可去除）→ **纯自动**六维门控加权评分（0-100 总分 + 维度画像），新方法（含传统 ML）丢 JSON 重跑即打分。

**用户已确认**：KW I-IV 级（病灶保留为权重重点）；总分+维度画像；现有人工分直接用+标注 N=6 局限；纯自动（新方法无需人工分）。

**实现要点**：
- 六维：D1 病灶保留 0.30 / D2 抗识破+分布 0.25 / D3 多样性 0.20 / D4 血管 0.10 / D5 颜色 0.08 / D6 记忆 0.07；公式本体 = `SCHEME` dict（报告逐字引用）
- 门控 `R=D2`：D1/D4/D5 乘 R；缺失键容错（维内归一 / 整维重归一，只跑通用层的新方法也能出分）；人工分缺失不影响打分仅不参与校准
- 接口：`python eval/score_scheme.py [--no-gate] [--models a,b] [--scorecard] [--datadir <dir>]` → 校准表 + 评分表 + `eval_data/_scores.json`（formula_version/weights/gate/tau/n_models）+ `--scorecard` 出 `figures/scorecard.png`；`plot_metrics.py` 有 `_scores.json` 时自动补第五张 `score_overview.png`
- 报告整合：5.6 指标-人工分校准分析（23 项 ρ 表 + 五矛盾解读 + 三重判据方法学）、5.7 自动综合评分（公式/权重/门控实证/总分表/限制声明）、**5.8 为什么这么设计（设计决策链）**、第二部分决策表加「评分层」行、第三部分数据文件补 _scores.json/score_scheme.py/scorecard
- **设计文档：docs/09-Score-Scheme-Design.md**（完整"为什么"：动机/六维权重理由/门控论证/归一化动机/指标去留判据/局限/新方法接入流程）；脚本 SCHEME 注释同步写理由（代码即文档）

**实测总分（门控后）**：film_l1lpips 72.9 / base_cj 44.2 / cond 44.0 / film 44.0 / vae 25.7 / dcgan 7.0；Kendall τ（vs 人工）= 0.600（N=6 小样本 + cond/film 同分并列 → τ_b 校正，参考为主）
- 无门控对照：film_l1lpips 76.1 / film 58.9 / base_cj 58.5 / **VAE 57.9** / cond 55.2 / dcgan 14.5——VAE 无门控混入中游正是门控必要性证据
- film 低于 cond/base_cj：人工-指标分歧簇，如实写进报告（诚实反映而非 bug）

**⚠️ 遇到的问题（回顾，供后续扩展参考）**：
1. **早期 Plan agent 的"门控 τ 0.80→0.93"不可信**：N=6 时 Kendall τ 只能取离散值（0.467/0.733/0.867/1.0），0.93 数学上不可能；其权重表/门控收益全部需实测复现，不可照搬
2. **门控的真实价值不是提 τ，而是语义正确性**：实测无门控 VAE=57.9（靠颜色/血管假象混到中游）→ 门控后 25.7（跌入失败带）
3. **指标-人工分歧簇（film/cond/base_cj）**：film 人工第 2 但客观分布/抗识破是 4 个扩散里最差（FID 200.2 / C2ST 0.989）——方案按指标排后与人工不一致，这是写进报告的洞察（扩增用途下人工偏爱多样性），不是 bug
4. **校准 ρ 实测（N=6，scipy spearmanr）**：KID -0.886、MS-SSIM -0.829、C2ST -0.771 与人工强吻合；Recall +0.580；LPIPS +0.200（方向反，VAE 悖论 → 出计分仅参考）；颜色 ρ≈-0.14（可被 VAE 博弈 → 门控+低权）；1-NN 全模型饱和 0.975-1.0 无区分 → 降权

**新方法接入**：评估 JSON 进 eval_data/ → `python eval/score_scheme.py --scorecard && python eval/plot_metrics.py` 即得总分/画像/全图（显示名在 plot_metrics.py `MODEL_LABELS` 补）；min-max 归一化是相对分数，模型库扩大后旧分数会平移（已写进报告限制声明）

### 4. 待用户确认的事项
- [x] **Phase A 自设计指标已补齐**（2026-08-01 晚：metrics_fundus.py 病灶/血管/记忆/C2ST/BRISQUE/Vessel Dice 全部跑完并写入报告）→ Bug 1 的"通用+自设计"两层指标均已落地
- [x] **Phase B 评分标准调研已完成**（2026-08-02：结论并入 evaluation_report.md 第二部分，"保留现有两层指标 + TSTR/TRTR 为终极金标准 + RETFound-FD 可选探索"）→ Bug 1 第二半落地
- [x] **老师已答复分类模型要求（2026-08-02）**：只做分级分类、330需标注、分级标准 = KW I-IV 级（老师说的 HR，用户已确认）→ 剩余：等 330 标注到位 + KW 分级细节确认 → 定 Phase C/D
- [ ] 是否有用户自己发现的报告/数据问题要改？

---

## 六、后续任务（Phase B/C/D）

### Phase B：评分标准调研（Bug 1 第二半）✅ 已完成 2026-08-02
结论已并入 `research-report/evaluation_report.md` 第二部分（评估报告与评分标准调研合并版）。核心结论：
- 现有两层指标方案经文献检验**无需推翻**（FID 小样本有偏但同量横向对比有效、KID 无偏、记忆检测独立于保真指标是共识）
- **新增决策**：TSTR/TRTR 定为终极金标准（Phase D 验收）；记忆检测升级为"必报"指标；FID 补小样本偏差说明
- **可选探索**（不阻塞）：RETFound-FD（域适配特征，需实测）、Betti-0/曲率血管指标（需先改善 conditions/ mask 质量）
- **明确不做**：域适配提取器重算全部历史模型（Woodland 证据：医学预训练不必然更好）

### Phase C：新方法探索（Bug 2）
- **C1 经典增广**（⏸ 用户 2026-08-02 表态：不用做，已报告过可用）——若老师后续要求可随时补跑
- **C2 传统 ML**（1 次实验，预期否决）：PCA/GMM/补丁合成，确证后作为否定性结论
- **C3 Stable Diffusion**（可选）：先调研 8GB VRAM LoRA 可行性，再跑

### Phase D：下游分类器验证（终极目标）
**老师答复后调整（2026-08-02）**：只做 DR 分级分类（纯分类网络，无检测）；330 张需标注等级（老师/医生标）；分级标准 = **KW（Keith-Wagener）高血压视网膜病变 I-IV 级**（用户 2026-08-02 确认老师说的 HR 即此，本项目生成 IV 级最重）→ 分类器类别数 = KW 等级数（等 KW 分级细节确认）。剩余路径：等 KW 分级细节 + 330 标注 → 定标签格式 + 公开数据对齐策略（找 KW 标注公开集 or 将公开集重标为 KW）→ ResNet/EfficientNet 分类器 → TSTR/TRTR 验收"合成重度图显著提升对应类 Recall/F1/平衡准确率"。

---

## 七、输出目录速查

| 目录 | 内容 |
|------|------|
| `docs/` | 工程规范 9 份（00 导航+路径索引 / 01 需求 / 05 评估体系 / 06 路线 / 08 工作说明） |
| `research-report/` | REPORT（原版）.docx（+ _with_metrics 待替换版）/ 本指引 CONTINUE_GUIDE_NEW.md / **evaluation_report.md/.docx**（评估+调研合并版）/ interview-prep / 临时脚本 `_insert_metrics.py`（待删） |
| `eval/` | **metrics_common.py**（通用层）+ **metrics_fundus.py**（专用层自设计）+ **plot_metrics.py**（可视化，自动扫描 JSON 可复用）+ **score_scheme.py**（综合评分：六维门控 0-100 总分 + 人工分校准）+ **README.md**（用法说明） |
| `research-report/figures/` | 评估可视化四图（general_metrics / color_metrics / fundus_metrics / radar.png）+ scorecard.png（综合评分卡片）+ score_overview.png（综合评分总览，plot_metrics 自动补），已整合进 evaluation_report.md |
| `eval_data/` | 评估图（real/ 330 张 + 每模型 300 张，**已 .gitignore 只提交 JSON**）+ 各模型 `*_metrics.json` / `*_fundus_metrics.json` + `_step_compare/`（50/100 步 DDIM 对比图）+ `_scores.json`（综合评分输出） |
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/results_film_l1lpips/` | 最佳模型结果（checkpoint: models/final_model.pth） |
| `generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions/` | 330 张血管骨架 mask（条件+评估用） |
| `fundus/` | `_all_images_ORIGINAL/`（330 张真实眼底图，训练基准 + real 集）+ **rotate_augment_check.py** + **使用说明.txt**（旋转安全筛选工具） |

## 八、快速命令

```bash
# 推送到双远端
git push github main && git push gitee main

# GPU 确认
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 评估通用层单模型（在项目根目录）
python eval/metrics_common.py --real eval_data/real --fake eval_data/<model>/singles --img_size 128 --device cuda --json

# 评估专用层自设计指标（条件模型加 --cond_path；--skip_c2st 可跳过 CNN 训练）
python eval/metrics_fundus.py --real eval_data/real --fake eval_data/<model>/singles \
    --model <model> --device cuda [--cond_path generate_project/deep_learning/Fundus-Diffusion/ddpm/conditions] [--skip_c2st] --json

# 评估可视化（自动扫 eval_data/ 全部 JSON → research-report/figures/ 四图；新方法重跑即入图）
python eval/plot_metrics.py

# 综合评分 + 人工分校准（校准表 + _scores.json + scorecard.png；新方法丢 JSON 重跑即打分）
python eval/score_scheme.py [--no-gate] [--scorecard]

# 生成评估图（在 generate_project/deep_learning/Fundus-Diffusion/ddpm/ 下，串行！8GB 不能并行）
python generate.py --checkpoint <ckpt> --num_images 300 --output_dir ../../../eval_data/<model> \
    --grid_size 0 --sampler ddim --sampling_steps 50 \
    --base_dim <64|128> --dim_mults 1 2 3 4 --attn_layers 2 \
    [--cond_path ./conditions] --seed 42
```
