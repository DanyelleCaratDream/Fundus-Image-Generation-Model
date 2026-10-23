# SD21-LoRA —— Stable Diffusion 2.1 + LoRA（B1 线）

> `pretrained/` 下的**具体方法族**（对标 `Fundus-GAN/DCGAN` 的组织方式）。
> 回答的问题：**借大规模眼底预训练模型，能不能用 72 张 severe 图注入病理、得到可交付的 B 域生成器。**

## 1. 方法标注

| 维度 | 选择 |
|:--|:--|
| 底座 | Stable Diffusion 2.1-base（`LanguageMachines/stable-diffusion-2-1-base`） |
| 外部先验 | KylianSu 眼底 LoRA（rank 64，diffusers 格式，作者私有眼底数据训 36k 步；**只学正常眼底**） |
| 我们的微调 | peft LoRA（rank 32 / alpha 32，UNet 的 `to_k/to_q/to_v/to_out.0`，6.64M 可训练参数） |
| 训练数据 | 老师 severe train 唯一底图 **72 张**（512×512，跳过 `_aug_1`） |
| 目标 | 补丁化 severe LoRA（B1b）；与外部 LoRA 加权合并（B1c）；img2img 低强度精修（B1d） |

> 这不是新框架，而是**预训练扩散 + 参数高效微调**；与自研 Diffusion/CFM 线的区别是
> “先验来自外部大规模数据”，代价是病理信息只能从 72 张里学。

## 2. 结论（先看这个）

- **B1a 零训练**：8/8 结构真实的**正常**眼底，血管细节好，但**无病灶**；色彩可用 Reinhard LAB 校正。
- **B1b 500 步**：首次出现病灶样结构（出血斑/渗出/金黄颗粒），但作者判“仍一眼生成、视盘过大、白斑不显”。
- **B1c 合并**：三组配比均劣于不合并——外部 LoRA 范数占优，把病灶和色调一起稀释。
- **路线 A（patch 复训 216 块）**：白斑出现但结构塌陷（盘内标准差 27.6→12.5–14.5），判死。
- **B1d 精修 0.25**：局部细节改善，但会幻觉式添加（个别图两个视盘），只能当后处理。
- **总评**：预训练给真实感、不给病理；真实感仍不如自研 D3/CFM 线。B1 保留为对照。

## 3. 目录结构

```
SD21-LoRA/
├── README.md
├── train/b1b_lora_train.py          # severe LoRA 训练（温度守护、每 125/250 步存档）
├── infer/
│   ├── b1a_lora_infer.py            # SD2.1 + KylianSu LoRA 纯推理（B1a）
│   ├── b1b_lora_infer.py            # 加载自训 peft LoRA（B1b）
│   ├── b1c_merge_infer.py           # 双 LoRA ΔW 加权合并（B1c）
│   └── b1d_refine_infer.py          # KylianSu img2img 低强度精修（B1d）
├── tools/
│   ├── b1a_color_transfer.py        # Reinhard LAB 色彩迁移
│   ├── b1b_make_data.py             # 重建 512 训练数据
│   ├── b1b2_make_data.py            # patch 裁剪（路线 A）
│   ├── *sheet*.py                   # 各种同槽对比图
│   └── b1f_head_to_head.py          # B1 vs D3 vs CFM 五方对决（跨线）
├── data/b1b_data/                   # severe512（72）+ severe_patches（216）
├── results/b1b_lora/                # 正式检查点 lora_step_00500.safetensors（权重不入 git）
├── results/probe_b1a/               # B1a 探针网格
└── judgment/                        # B1a/B1b 单图、对比总览、五方对决
```

## 4. 运行

```bash
# 重建训练数据（源：fundus/HRSG_4_758_square/train/severe，只读）
python tools/b1b_make_data.py

# 训练（默认 1500 步，每 250 步存档；500 步为最佳）
python train/b1b_lora_train.py --steps 1500 --batch 2

# 出图
python infer/b1b_lora_infer.py --lora results/b1b_lora/lora_step_00500.safetensors \
    --num 8 --seed 42 --steps 100 --out judgment/b1b_500

# 精修
python infer/b1d_refine_infer.py --src judgment/b1b_500/singles --strengths 0.25
```

## 5. 注意

- 权重（`*.safetensors`）按根 `.gitignore` 不入 git；数据目录可由脚本重建；
- `b1c_merge_sheet.py` / `b1b2_scan_sheet.py` 引用的中间图已在清理时删除，仅作历史脚本保留；
- KylianSu LoRA 许可 TBD，论文表述用“采用/借鉴其范式”，需联系作者确认。
