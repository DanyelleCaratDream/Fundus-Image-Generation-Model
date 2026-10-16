#!/usr/bin/env bash
# 合并数据集从零训练（2026-10-16 夜）
#
# 背景与目的：
#   · DL-EX-015（76张@256, base_dim=64）失败 —— 生成糊块
#   · DL-EX-016（76张@128, base_dim=64）同样失败 —— 排除「分辨率」假说
#   · 两个失败实验的 loss 都正常收敛 → 倾向「数据量不足」
#   · 现在把数据从 76 张扩到 239 张（+老师 severe 76 + A域原始 163）
#     —— **只改「数据量」这一个变量**，其余超参与 DL-EX-015 完全一致
#
# 如果这次能出图 → 证实「瓶颈是数据量」；如果还是不行 → 假说被推翻，得重新找原因。
set -u
cd "$(dirname "$0")"

DS=../../../../fundus/combined_sq256
CD=./conditions_combined
OUT=results_D_combined239_scratch
LOG=/tmp/combined_scratch.log

log() { echo; echo "=============================================================="; \
        echo "[$(date '+%F %T')] $*"; echo "=============================================================="; }

log "从零训练：合并数据集 239 张 @256（base_dim=64, batch=2, 500轮）"
echo "  数据: $DS（$(ls $DS | wc -l) 张）"
echo "  条件: $CD（$(ls $CD | wc -l) 张 mask）"
echo "  输出: $OUT"
echo

# 等 GPU
w=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 1500 ] && { echo "  GPU 空闲（${used} MiB）"; break; }
    sleep 30; w=$((w+30))
    [ $((w % 300)) -eq 0 ] && echo "  等待 GPU 空闲 ${w}s"
    [ "$w" -gt 900 ] && { echo "  等待超时"; exit 1; }
done

rm -rf "$OUT"
python -X utf8 train.py \
    --dataset_path "$DS" --cond_path "$CD" --output_dir "./$OUT" \
    --img_size 256 --base_dim 64 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 500 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 > "$LOG" 2>&1

log "训练结束，用原始权重出 8 张评估图"
rm -rf /tmp/evalraw_D
python -X utf8 generate.py --checkpoint "$OUT/models/final_model.pth" \
    --num_images 8 --output_dir /tmp/evalraw_D --grid_size 0 \
    --img_size 256 --base_dim 64 --dim_mults 1 2 3 4 --attn_layers 2 \
    --sampler ddim --sampling_steps 100 --use_raw \
    --cond_path "$CD" > /dev/null 2>&1
echo "  → /tmp/evalraw_D/singles（$(ls /tmp/evalraw_D/singles 2>/dev/null | wc -l) 张）"

log "全部完成"
