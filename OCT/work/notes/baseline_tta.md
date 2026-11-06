# TENT / CoTTA 基线实现说明

> 目的：记录这两个基线的**作用对象、实现取值与已声明的偏离**，作为复现报告中"基线"一节的依据。
> 代码：`repro/code/baselines/tta_seg.py`（适配器）、`repro/code/baselines/eval_tta_seg.py`（驱动）
> 测试：`repro/code/tests/test_tta_seg.py`（18 项，全部通过）、`tests/test_eval_tta_pipeline.py`（链路自测）

## 1. 关键理解：它们不是重建方法

论文第 2 节把 TENT [24] / CoTTA [25] 描述为
"early methods updated pre-trained model weights during inference using target domain data
and specialized loss functions"，第 3 节列为对照方法。两条独立证据表明它们**作用于下游分割模型**：

1. **表格结构**：TENT/CoTTA 行的 FID 列为 `N/A`（Topcon 块的 GA 列亦为 `N/A`）——不产生图像，
   自然无法计算 FID；
2. **方法语义**：二者的机制是"用无标签目标域数据在线更新模型权重"，没有任何图像生成过程。

因此本实现的作用对象与数据流是：

| 方法 | 输入图像 | 被更新的模型 | 与论文表格的对应 |
|---|---|---|---|
| Baseline | Cirrus/Topcon 原始图 | 不更新 | Baseline 行 |
| TENT | Cirrus/Topcon 原始图 | 分割模型（仅 BN 仿射参数） | TENT 行 |
| CoTTA | Cirrus/Topcon 原始图 | 分割模型（全部参数）+ EMA 教师 | CoTTA 行 |
| Ours | **重建后**的图像 | 不更新（同一份分割权重） | Ours 行 |

四行共用**同一份分割模型权重**与**同一批测试切片**，差异才唯一归因于适配策略。

> ⚠️ 若把 TENT/CoTTA 接到重建结果上，方法就变质了，与论文对照含义不同。

## 2. 实现要点

### TENT
- 只更新归一化层仿射参数（`scope=bn_affine`），损失为预测熵。
- 多标签 sigmoid 头的熵取**各通道二值熵之和**——这正是各通道独立伯努利分布的联合熵，
  与 TENT 在单标签 softmax 上的定义同构（`binary_entropy`，已用手算样例 `3·ln2` 验证）。
- 归一化层使用**测试批统计量**（TENT 原论文设定）。实现上只切换归一化层的模式，
  **不**把整个模型设为 `train()`，以免重新打开 dropout 等训练态行为。
- 源域 running stats 全程不被污染（有测试守护）——否则同一 checkpoint 会随评测顺序漂移。

### CoTTA
- 教师-学生结构：教师由源域模型深拷贝得到，学生负责更新，教师 EMA 跟随（α=0.999）。
- 伪标签：对 `n_aug` 个增强视图取教师预测的平均，按置信度阈值 τ=0.5 二值化。
- 类别均衡队列：每类一个队列，选"该类前景像素平均置信度最高"的类别入队；
  队列满则弹出最旧条目、与当前条目一起参与学生更新。
- 随机回退（stochastic restore）：按概率 p=0.01 把教师参数逐元素回退到源域值。
- 预测使用**教师**模型（原论文设定）。

## 3. 已声明的实现选择与偏离

论文仅说明"Hyperparameters for all methods were chosen by grid search"，**未给出具体取值**，
因此以下取值需在报告中记为实现选择而非论文复刻：

| 项目 | 取值 | 依据 |
|---|---|---|
| TENT lr / momentum / 每批步数 | 1e-3 / 0.9 / 1 | TENT 原论文 ImageNet-C 默认 |
| CoTTA EMA α / 回退概率 p / 阈值 τ / 队列长度 | 0.999 / 0.01 / 0.5 / 100 | CoTTA 原论文默认值 |
| 增强个数 | 2 | 计算成本与稳定性的折中，可经 `--cotta-n-aug` 调整 |
| 更新范围 | TENT=BN 仿射；CoTTA=全部参数 | 各自原论文设定，可用 `--tent-scope/--cotta-scope` 覆盖 |

**明确偏离项**：

1. **增强算子替换**：原版 CoTTA 面向 RGB 自然图像，使用 color jitter / Gaussian blur /
   random affine。OCT 为单通道灰度图，色相与饱和度无意义，故只保留**亮度与对比度抖动**
   （±20%/±10%），并保留高斯模糊与小幅仿射（±5°、±5% 平移、±5% 缩放，避免破坏细小病灶）。
   增强后取值被夹回输入自身的动态范围。
2. **队列条目粒度**：原论文以批为入队单位。这里同样是批级条目，但只保存被选中类别的
   硬伪标签（3 通道中取 1 通道），以把队列内存限制在可接受范围（≤ queue_size × 批张量）。
3. **学生监督方式**：对每个队列条目只监督其被选中的那个类别通道（BCE），
   而非三通道同时监督——与"类别均衡选择"的语义一致。

## 4. 一个需要如实披露的混淆项

论文的 Baseline 行使用源域 running stats，而 TENT/CoTTA **按各自方法设定**改用测试批统计量。
因此 Baseline → TENT/CoTTA 的差值同时包含两部分：(a) 权重适配；(b) 归一化统计量切换。
这是忠实复现各方法的必然结果（不是实现偏差），但解读差值时应说明。
代码已提供 `--tent-source-stats` 用于把 (b) 单独剥离出对照。

## 5. 工程细节

- **单样本批次**：批大小为 1 时批统计量退化，且当末端特征图缩到 1×1 时 PyTorch 的
  BatchNorm 会直接报错。丢弃该批次会**静默改变评测切片集合**，因此实现对单样本批次
  临时回退到源域 running stats，并把发生次数记入结果字段
  `singleton_batches_using_source_stats`。
- **数据流协议**：默认按 CSV 顺序逐体积流入（贴近真实临床序列）；
  `--shuffle` 切换为 i.i.d. 协议。`--reset-each-batch` 切换为 episodic 协议。
- **无标签泄漏**：目标域标签只用于最后的 Dice 计算，从不参与适配。

## 6. 运行方式

```bash
# Baseline（应与 eval_dice.py --raw-dir 结果逐位一致，自测已验证）
python baselines/eval_tta_seg.py --csv retouch_cirrus.csv --ckpt runs/seg/fold0/best.pt \
    --mode none  --out results/baseline_cirrus.json

# TENT
python baselines/eval_tta_seg.py --csv retouch_cirrus.csv --ckpt runs/seg/fold0/best.pt \
    --mode tent  --out results/tent_cirrus.json --batch-size 16

# CoTTA
python baselines/eval_tta_seg.py --csv retouch_cirrus.csv --ckpt runs/seg/fold0/best.pt \
    --mode cotta --out results/cotta_cirrus.json --batch-size 16
```

## 7. 对照目标（论文 Table 1，Ours 之外的对照行）

| 行 | Cirrus→Spectralis（IRF/SRF/PED/Mean） | Topcon→Spectralis |
|---|---|---|
| Baseline | 51.7 / 43.2 / 22.5 / 39.1 | 49.9 / 42.3 / 28.0 / 40.0 |
| TENT | 51.5 / 64.4 / 35.5 / 50.5 | 53.9 / 47.9 / 33.3 / 45.0 |
| CoTTA | 52.0 / 64.5 / 36.5 / 51.0 | 53.5 / 47.7 / 32.7 / 44.6 |

## 8. 验证状态（2026-10-24）

| 检查 | 结果 |
|---|---|
| `tests/test_tta_seg.py` 单元测试 | ✅ 18/18 通过 |
| `tests/test_eval_tta_pipeline.py` 链路自测 | ✅ 2/2 通过 |
| 交叉验证：`--mode none` vs `eval_dice.py --raw-dir` | ✅ 逐位一致（最大差 0.0e+00） |
| `tests/test_seg_lib.py` | ✅ 11/11 通过 |
| `tests/test_flow_pipeline.py`（官方仓库路径回归） | ✅ 6/6 通过 |
| 源码完整性检查 | ✅ 13/13 文件通过 |

> 以上均为**代码正确性**验证；真实数值仍需 RETOUCH 数据到位后才能产出。