#!/usr/bin/env bash
# 2026-10-16 夜：合并数据集从零训练，两组对照
#
# 目的：验证「数据量是瓶颈」这个假说
#   D1 = 193张 @256, base_dim=64   ← 与失败的 DL-EX-015（76张@256, bd64）**只差数据量**
#   D2 = 193张 @256, base_dim=128  ← 在 D1 基础上再加容量，检验「容量不足」假说
#   （数据 2026-10-16 已去重+去泄漏：239 → 193，见 EXPERIMENTS.md DL-EX-019 与 fundus/_removed_dups_sq256/）
#
#   若 D1 成功 → 证实瓶颈是数据量（D2 用来确认容量还有没有额外收益）
#   若 D1 失败、D2 成功 → 容量才是关键
#   若都失败 → 两个假说都推翻，得重新找原因
#
# 注意 D2 会溢出显存（256px + bd128 + batch2 实测 8.64GB > 8GB），速度会慢不少，
# 所以 D2 只跑 300 轮——验证容量假说不需要跑满。
set -u
cd "$(dirname "$0")"

DS=../../../../fundus/combined_sq256
CD=./conditions_combined

log() { echo; echo "=============================================================="; \
        echo "[$(date '+%F %T')] $*"; echo "=============================================================="; }

wait_gpu() {
    local w=0
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        [ -n "$used" ] && [ "$used" -lt 1500 ] && { echo "  GPU 空闲（${used} MiB）"; return 0; }
        sleep 30; w=$((w+30))
        [ $((w % 300)) -eq 0 ] && echo "  等待 GPU 空闲 ${w}s"
        [ "$w" -gt 900 ] && return 1
    done
}

eval_raw() {  # $1=ckpt $2=base_dim $3=tag
    rm -rf "/tmp/evalraw_$3"
    python -X utf8 generate.py --checkpoint "$1" --num_images 8 \
        --output_dir "/tmp/evalraw_$3" --grid_size 0 \
        --img_size 256 --base_dim "$2" --dim_mults 1 2 3 4 --attn_layers 2 \
        --sampler ddim --sampling_steps 100 --use_raw --cond_path "$CD" > /dev/null 2>&1
    echo "  → /tmp/evalraw_$3/singles（$(ls /tmp/evalraw_$3/singles 2>/dev/null | wc -l) 张）"
}

echo "  数据: $DS（$(ls $DS | wc -l) 张）"
echo "  条件: $CD（$(ls $CD | wc -l) 张）"

wait_gpu || { echo "  GPU 等待超时"; exit 1; }

# ==================== D1：base_dim=64 ====================
log "D1：193张 @256, base_dim=64, batch=2, 500轮"
rm -rf results_D1_combined239_bd64
python -X utf8 train.py --dataset_path "$DS" --cond_path "$CD" \
    --output_dir ./results_D1_combined239_bd64 \
    --img_size 256 --base_dim 64 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 500 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 > /tmp/D1.log 2>&1
log "D1 训练结束"
eval_raw results_D1_combined239_bd64/models/final_model.pth 64 D1

# ==================== D2：base_dim=128 ====================
log "D2：193张 @256, base_dim=128, batch=2, 300轮（会溢出显存，慢）"
rm -rf results_D2_combined239_bd128
python -X utf8 train.py --dataset_path "$DS" --cond_path "$CD" \
    --output_dir ./results_D2_combined239_bd128 \
    --img_size 256 --base_dim 128 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 300 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 > /tmp/D2.log 2>&1
log "D2 训练结束"
eval_raw results_D2_combined239_bd128/models/final_model.pth 128 D2

log "D1 / D2 全部完成"
