# code/ —— 外部实现与我们的实现

## 1. 我们的 CFM 实现（就在现有仓库里，不重复造）

**位置**：`../../Fundus-Diffusion/ddpm/train.py`（训练）与 `generate.py`（采样）。

关键函数：
| 函数 | 行 | 作用 |
|:--|--:|:--|
| `cfm_path(x1, t_cont, noise=None)` | 385 | OT 线性路径：`x_t=(1-t)x0+t x1`，`v=x1-x0` |
| `cfm_predict_x1(x_t, t, v_pred)` | 400 | 由速度反推数据 `x1 = x_t + (1-t)v`，供 LPIPS/颜色正则 |
| `euler_sample(model, shape, steps, cond)` | 406 | Euler 积分采样，NFE=steps |
| `t_cont_to_index(t_cont)` | 380 | 连续 t → 离散 time embedding 下标 |

启用方式（无需改代码）：
```bash
F:/python/python.exe train.py --objective cfm \
    --loss_type l1 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --dataset_path ../../../../fundus/combined_sq128 \
    --cond_path ./conditions_combined_128 \
    --resume ./results_film_l1lpips/models/final_model.pth \
    --img_size 128 --base_dim 128 --batch_size 2 --epochs 1100 \
    --output_dir ../Fundus-FlowMatching/results/cfm128
```
采样：`generate.py --sampler auto`（读到 cfm checkpoint 自动用 Euler）。

> 现状（2026-10-23）：已训练两条线——`results/cfm128/`（320 轮，loss 0.12–0.14）与
> `results/cfm256/`（原生 256，loss 0.100，作者目检 97/100）。见 ../README.md 第 2 节。
> 路径说明：训练器在 `Fundus-Diffusion/ddpm/`，实验输出按约定放本目录 `results/`。

## 2. 外部库（code/external/）

| 目录 | 仓库 | 用途 | 安装 |
|:--|:--|:--|:--|
| `external/flow_matching/` | [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | Meta 官方库：连续/离散 FM、条件路径、OT 耦合 | `pip install -e external/flow_matching` |
| `external/torchcfm/` | [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) | OT-CFM / SB-CFM / [SF]²M 等变体 | `pip install torchcfm` |

未克隆（需要时再下）：
| 仓库 | 用途 | 地址 |
|:--|:--|:--|
| MOTFM | 医学 mask 条件 OT-FM 参考实现 | https://github.com/milad1378yz/MOTFM |
| RectifiedFlow | reflow 官方实现 | https://github.com/gnobitab/RectifiedFlow |
| SD3 generative-models | 大规模 RF 实现 | https://github.com/Stability-AI/generative-models |
| awesome-flow-matching | 论文索引 | https://github.com/dongzhuoyao/awesome-flow-matching |

> `external/` 不入 git（体积 + 上游变动）；需要时按上表重克隆即可。

## 3. 不建议做的事

- **不要**把 `train.py` 的 CFM 分支复制一份到本目录——两份代码会分叉。
  要改造就改 `Fundus-Diffusion/ddpm/train.py`，本目录只放实验配置与结果。
- **不要**引入第二个训练框架（TorchCFM 只作参考/对照实现，不当主训练器）。
