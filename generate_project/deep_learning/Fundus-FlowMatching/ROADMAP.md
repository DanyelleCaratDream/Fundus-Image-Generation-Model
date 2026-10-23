# Flow Matching 实验路线图（按性价比排序）

> 作者 2026-10-23 确认：**后面就按这个顺序跑**。每完成一项，先问作者是否记入 DLreport
> （见 memory: dlreport-update-protocol）。
>
> 排序原则：先用零训练/低成本实验拿到可判断的信号，再投入 5 小时级的重训，
> 最后才是 1–2 天级的改造。每项都给出“要回答什么问题”和“看什么指标”。

## 总表

| # | 实验 | 借鉴来源 | 成本 | 要回答的问题 | 状态 |
|:--|:--|:--|:--|:--|:--|
| 1 | **轨迹对齐直方图匹配** | TTA-Flow | 0 训练，约 10 分钟 | 推理期色彩/域对齐能否替代或增强 Reinhard？ | 工具已实现，等 GPU |
| 2 | **时间语义对齐消融**（cfm128_align） | 我们自己的发现 | 320 轮 @128，约 5h | 时间轴反转是否拖慢了 DDPM→CFM 迁移？ | 代码已实现，排队 |
| 3 | **去时间条件消融** | TTA-Flow / He et al. ICML25 | 320 轮 @128，约 5h | 完全去掉 t 条件是否更稳、更好？ | 未实现（需改 UNet） |
| 4 | **x1 参数化消融** | TTA-Flow | 320 轮 @128，约 5h | 直接预测 x1（速度反推）是否优于预测 v？ | 未实现 |
| 5 | **检索 + 重心条件** | FSFM | 1–2 天 | 72 张 severe 上做类内插值能否提升多样性？ | 未实现（需编码器） |
| 6 | **双向流统一生成/分类** | SymmFlow | 中期 | 能否把 severe 生成与 DR 分级统一进一个模型？ | 仅参考 |

## 每项的判据与产出

### 1. 轨迹对齐直方图匹配
- 实现：`CFM/tools/tta_hist_match.py`（参考轨迹平均直方图 → 测试图 CDF 匹配 → 从 s_target 启动 Euler）。
- 判据：颜色距离（vs 真实 severe）、病灶保留率、目检；与 Reinhard 做 A/B。
- 产出：`CFM/results/tta_hm/`。

### 2. 时间语义对齐（cfm128_align）
- 设置：同 cfm128（初始化/数据/轮数/种子），唯一差异 `--cfm_time_align`。
- 判据：前 25 轮 loss 下降速度、收敛 loss、8 张同种子出图、FID/BRISQUE/颜色距离。
- 产出：`CFM/results/cfm128_align/`。

### 3. 去时间条件
- 设置：改 UNet，不输入时间嵌入（参考 TTA-Flow 的 UNet）；同 cfm128 预算。
- 判据：训练稳定性（是否还能收敛）、细节/病灶指标、与 cfm128_align 对照。
- 产出：`CFM/results/cfm128_notime/`。
- 注意：这是架构改动，不是开关；需先写一个 `UNetNoTime` 或 `--no_time_cond` 分支。

### 4. x1 参数化
- 设置：网络输出 x1，速度由 `(x1−z_t)/(1−t)` 恢复（TTA-Flow Eq.4）；同预算。
- 判据：loss 收敛、LPIPS/颜色正则是否更直接、出图质量。
- 产出：`CFM/results/cfm128_x1/`。

### 5. 检索 + 重心条件（FSFM）
- 设置：冻结编码器（RETFound 或我们的 A 域模型），取 anchor 的 k 近邻，Dirichlet 权重组合条件。
- 判据：LPIPS 多样性、下游分类（TSTR）、目检。
- 产出：`CFM/results/cfm128_retrieval/`。
- 风险：需要编码器与额外数据管线，收益主要在多样性而非病理保真。

### 6. 双向流（SymmFlow）
- 设置：图像↔标签联合流；替代 A2 分类器。
- 判据：生成质量 + 分类 QWK。
- 产出：`SymmFlow/results/symmflow_fundus/`。
- 风险：193 张数据量可能不足；改造量大。

## 预训练对照（作者 2026-10-23 追加，两项都做）

| # | 实验 | 做法 | 成本 | 状态 |
|:--|:--|:--|:--|:--|
| 7 | **Tiger-Model 零样本/迁移观察** | 官方权重（pretrain + controlnet_bg，6.2GB）已下载；文本提示直接生成 | 推理约 1h | 权重就绪、脚本就绪（`pretrained/TextGuidedDiffusion/tools/tiger_zeroshot.py`），等 GPU；它是**超声图**训练的，域差大，预期负结果 |
| 8 | **轻量文本条件 A/B** | 在 CFM 的 UNet 时间嵌入上加零初始化 CLIP 文本投影（normal/severe 两个提示），与 cfm128 同预算对比 | 320 轮 @128，约 5h | **代码已实现**（`--use_text_cond`，预设 `cfm128_text`），排队 |

> 这两项回答的是“文本/预训练能带来什么”，与 1–6 的 Flow Matching 族内消融并列；
> 完成后先问作者是否写入 DLreport。

## 无人值守队列（2026-10-23 深夜启动）

`run_queue.py` 已启动，顺序：等 lesion 训练结束 → 评估 lesion → 第 1 项 → 第 7 项
→ 第 2 项训练+评估 → 第 8 项训练+评估。日志 `RUN_QUEUE.log`，状态 `RUN_QUEUE_STATUS.json`。
单项失败不阻塞后续；第 5 项（检索条件）等作者决定编码器后再跑。

## 依赖与限制

- GPU 单卡 8GB：同一时间只能跑一个训练；推理类实验（#1）可与训练错开。
- 温度纪律 ≤70°C：训练器已内置每 25 步守护。
- 每项完成后：更新 `EXPERIMENTS.md`（执行记录），并按协议询问是否写入 DLreport。
