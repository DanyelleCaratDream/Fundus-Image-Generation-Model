# 官方仓库语义核对笔记（TTA-Flow）

> 核查对象：`https://github.com/Veit21/tta-flow`，commit `c2f11904e5c571c737e68ef209de09b3b93d0a04`
> 核查方式：逐行阅读 `src/test.py`、`src/util/flow_matching_util.py`，非推测
> 用途：本文件是复现设定的**唯一事实来源**；凡论文正文与代码不一致处，一律以代码为准，并在报告中标注

---

## 1. ODE 积分起点的真实语义（标注为 R1 核心项）

论文正文的表述是"在 `s_target` 处初始化，只积分剩余步数"。**代码并非如此**。

`src/test.py` 中与 `s_target` 相关的全部代码只有两处：

```python
s_target = int(cfg["test_parameters"]["s_target"] * num_timesteps)     # 第 166 行
...
slice_d_matched = flow_matching_util.match_cumulative_cdf_batch(
    source=slice_d,
    template_batch=reference_trajectories_np[:, s_target, 0, 0, ...]
)                                                                      # 第 195-197 行
...
inference_trajectory = ode_solver.solve(
    x=slice_d,
    t_span=torch.linspace(0, 1, num_timesteps, device=device),         # 第 205-208 行
)
pred_slice = inference_trajectory[-1]
```

结论：

| 项目 | 论文表述 | 代码实际行为 |
|---|---|---|
| `s_target` 的作用 | 积分起点 | **仅用于从参考轨迹数组里挑选第 `s_target` 步的平均直方图** |
| 积分区间 | `[s_target, 1]` | **恒为 `[0, 1]`**（`linspace(0, 1, num_timesteps)` 硬编码，与 `s_target` 无关） |
| 积分步数 | 剩余步数（随 `s_target` 变化） | 恒为 `num_timesteps - 1` 步 |

即：测试图先被**直方图匹配**到"模型样本在第 `s_target` 步时的强度分布"，再把这个匹配后的图像**当作 t=0 的初始条件**跑完整积分。

**为何这仍然可能是合理设计**：直方图匹配只改写强度分布、保留空间结构，相当于给测试图注入"第 `s_target` 步的对比度/幅值统计"，让起点在幅值尺度上接近模型在该时刻的样本；随后由完整 ODE 负责结构层面的精修。由于论文主推的 `Ours_uncond.` 是**时间无关网络**（`remove_noise_conditioning=True`），网络本身不含 t，ODE 仅通过 x-prediction 的重参数化项 `(x̂ − x_t)/clamp(1−t, min=0.05)` 显式依赖 t，因此从 t=0 起步与从 `s_target` 起步会产生不同轨迹，但差异来源明确、可控。

**复现处理**：两种设定都跑（`--ode-start from_zero`（代码原样）与 `--ode-start from_s_target`（论文表述）），以能对上论文数值者为准，并把该不一致显式记入复现报告。

---

## 2. `s_target` 的索引映射

```python
s_target = int(cfg["test_parameters"]["s_target"] * num_timesteps)
```

* `reference_trajectories_np` 形状为 `[N, T, 1, 1, H, W]`，`T = num_integration_steps`，时间点为 `linspace(0, 1, T)`。
* 因此下标 `s_target` 对应的**真实时间比例**是 `s_target / (T − 1)`，而不是 `s_target / T`。
* 例：`T = 51`、配置 `s_target = 0.8` → `int(0.8 × 51) = 40` → 实际时间比例 `40/50 = 0.80`。
  `T = 101`、`s_target = 0.8` → `int(80.8) = 80` → `80/100 = 0.80`。
* 结论：`int()` 截断与 `T` / `(T−1)` 的偏差在大 `T` 下基本相抵，**索引语义可用**；但当 `s_target` 取极端值或 `T` 较小时需单独核算。复现时以 `s_target/(T−1)` 反算真实时间比例并在报告中标注。

---

## 3. 直方图匹配的作用范围

```python
for idx, batch in enumerate(tqdm(test_dataloader, desc="Processing Volumes")):
    _D = batch.shape[1]                     # 该 volume 的 B-scan 数
    for d in tqdm(range(_D), desc="Iterating slices", leave=False):
        slice_d = batch[:, d:d+1, :, :]     # [1, 1, H, W]
```

* 匹配是**逐 B-scan** 进行的（不是逐 volume）。
* 模板是 `reference_trajectories_np[:, s_target, 0, 0, ...]`，形状 `[N, H, W]`，即 **N 条参考轨迹同一步的平均直方图**。
* `num_reference_trajectories == 0` 是**关闭直方图匹配的开关**（代码显式分支，日志打印 `Histogram matching deactivated!`）。这正是论文 Table 1 中 **FM [14]** 基线行的构造方式，无需额外实现。
* `num_references < 0` 抛 `ValueError`；非整数抛 `TypeError`。

---

## 4. 值域约定（曾误判，已按源码重新核实）

**关键更正**：磁盘上的 `.npy` 体积**不是** `[-1, 1]`，而是 **`[0, 1]`**。判据来自仓库的变换配置本身。

| 环节 | 值域 | 依据 |
|---|---|---|
| 磁盘 `.npy`（训练用体积） | **`[0, 1]`** | 由下一行反推：变换要求输入为 `[0, 1]` |
| 送入 flow matching 网络的张量 | `[-1, 1]` | `transforms_fm.yaml` 的 `Normalize(mean=0.5, std=0.5, max_pixel_value=1.0)`，计算 `(x − 0.5·1) / (0.5·1) = 2x − 1` |
| 模型输出（轨迹终点） | `[-1, 1]` | 生成模型的数据空间，先验为 `N(0, 1)` |
| **推理输出 `pred_*.npy`** | **`[0, 1]`** | `out_vol = out_vol.clip(-1, 1) / 2 + 0.5` |

**推导**：`Normalize(mean=0.5, std=0.5, max_pixel_value=1.0)` 的作用是 `(x − 0.5)/0.5`。
若输入 `x ∈ [0, 1]`，输出 `∈ [−1, 1]`，与高斯先验尺度一致；
若误把 `[-1, 1]` 的数据落盘，输出会被压成 `[−3, 1]`，尺度偏移达 2 倍且**不会报错**，只会让训练与推理同时跑偏。

**推论**：`.npy` 体积与 `pred_*.npy` 重建结果**值域相同**，因此下游分割模型可以无缝地先后处理二者——前提是统一采用同一约定。

**本项目对策**：`repro/code/downstream/seg_lib.py` 中统一以 **`[0, 1]`** 作为下游分割模型的输入约定，
训练与评测共用 `convert_range()`；值域来源由 `--source-range`、目标值域由 `--image-range` 显式声明，
并写入 checkpoint，评测时强制与训练一致（不一致直接报错，不静默）。

## 5. 推理输出文件命名与 CSV 行序

```python
for idx, batch in enumerate(tqdm(test_dataloader, desc="Processing Volumes")):
```

* dataloader 由 `FlowMatchingInferenceDataset` + `DataLoader(..., batch_size=1, shuffle=False)` 构造，因此 `idx` 就是**数据框（CSV）中的行序号**（0 起）。
* 输出文件命名为 `pred_{idx:03d}.npy`（`out_dir = <exp_dir>/inference/outputs`），**不携带体积名**。
* `repro/code/eval/eval_dice.py` 因此严格要求"`pred_*.npy` 文件数 == CSV 行数"，不一致时直接报错，避免序号错配产生无意义指标。

---

## 6. ODE 求解器的数值细节

```python
v_hat = (x_hat - x_current) / torch.clamp(1 - t_current, min=0.05)
```

* 采用 **x-prediction 参数化**：网络输出干净样本 `x̂`，再重参数化为速度 `v̂ = (x̂ − x_t)/(1 − t)`。
* 分母 `1 − t` 在 `t → 1` 时趋于 0，故 `clamp(min=0.05)` —— 即 **`t > 0.95` 之后的步长被统一按 `1/0.05 = 20` 倍放大**。这是刻意的数值保护，但会让最后若干步的更新量偏大，复现时不应"顺手修掉"。
* 支持 `euler` 与 `midpoint`（默认 `midpoint`）两种求解器；`sample_x` 由 `cfg["loss"]["target"] == "x"` 决定。
* `solve()` 返回**完整轨迹** `[T, B, C, H, W]`，推理只取 `[-1]`（终点）；中间点被丢弃（未利用轨迹本身做别的后处理）。

---

## 7. 由上述事实导出的复现设定清单

| 设定项 | 取值 | 依据 |
|---|---|---|
| 关推理 `histogram matching` 的基线 | `num_reference_trajectories = 0` | §3，仓库原生支持 |
| 时间无关版本 | `remove_noise_conditioning = True` | 论文主结果 `Ours_uncond.` |
| 有条件版本 | `remove_noise_conditioning = False` | 论文对照 `Ours_cond.` |
| 推理积分区间 | `[0, 1]` 全区间（代码原样） | §1 |
| 磁盘 `.npy` 值域 | `[0, 1]`（**不是 [-1,1]**） | §4 |
| 下游分割输入值域 | `[0, 1]`，训练与评测一致 | §4 |
| 重建结果与 GT 的配对 | 依据 CSV 行序，文件数严格校验 | §5 |
| DSC 统计口径 | 逐 B-scan 计算后取均值与标准差，仅统计被标注切片 | 论文 Table 1 的 ± 值量级（PED 可达 ±27）表明是切片级分布，非体积级 |