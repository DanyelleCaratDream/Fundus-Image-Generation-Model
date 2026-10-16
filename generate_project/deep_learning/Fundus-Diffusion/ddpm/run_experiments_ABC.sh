#!/usr/bin/env bash
# 2026-10-16 实验链：A -> C -> B
#
# 背景：2026-10-15 夜跑（76 张 train/severe @256, base_dim=64, 800轮）失败——
#       训练 loss 正常收敛、噪声预测优秀、能重建，但从纯噪声生成只得到紫蓝糊块。
#       同代码用 A 域模型（330张@128, base_dim=128）能出正常眼底图。
#       → 判定为数据量/建模难度问题，不是代码问题。
#
# A：把分辨率降回 128，其余不变 → 隔离"分辨率"这个变量
# C：分辨率保持 256，模型加宽到 128 → 隔离"模型容量"这个变量
# B：用 A 域最佳模型在 76 张上微调 → 检验"迁移学习"这条路
#
# 每个实验跑完都用 --use_raw 出一组图（EMA 在续训 checkpoint 上可能退化，不作为判据）
set -u
cd "$(dirname "$0")"

DS128=../../../../fundus/severe_train_sq128
DS256=../../../../fundus/severe_train_sq256
CD128=./conditions_severe_train_128
CD256=./conditions_severe_train

log() { echo; echo "=============================================================="; \
        echo "[$(date '+%F %T')] $*"; echo "=============================================================="; }

wait_gpu() {
    local w=0
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        [ -n "$used" ] && [ "$used" -lt 1500 ] && return 0
        sleep 60; w=$((w+60))
        [ $((w % 600)) -eq 0 ] && echo "   等待 GPU 空闲 ${w}s（当前 ${used:-?} MiB）"
        [ "$w" -gt 3600 ] && { echo "   等待超时"; return 1; }
    done
}

# 用原始权重出 4 张图，存到 <out>/eval_raw/
eval_raw() {  # $1=ckpt $2=img_size $3=base_dim $4=cond_path $5=dataset $6=out_tag
    rm -rf "/tmp/evalraw_$6"
    python -X utf8 generate.py --checkpoint "$1" --num_images 4 \
        --output_dir "/tmp/evalraw_$6" --grid_size 0 \
        --img_size "$2" --base_dim "$3" --dim_mults 1 2 3 4 --attn_layers 2 \
        --sampler ddim --sampling_steps 100 --use_raw \
        --cond_path "$4" --dataset_path "$5" > /dev/null 2>&1
    echo "   原始权重出图: $(ls /tmp/evalraw_$6/singles 2>/dev/null | wc -l) 张 -> /tmp/evalraw_$6/singles"
}

wait_gpu || exit 1

# ============ A：76 张 @128，base_dim=64（只改分辨率） ============
log "实验 A：76张 @128, base_dim=64, batch=2, 800轮（从零训）"
rm -rf results_A_severe128_scratch
python -X utf8 train.py --dataset_path "$DS128" --cond_path "$CD128" \
    --output_dir ./results_A_severe128_scratch \
    --img_size 128 --base_dim 64 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 800 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 2>&1 | tail -3
eval_raw results_A_severe128_scratch/models/final_model.pth 128 64 "$CD128" "$DS128" A128

# ============ C：76 张 @256，base_dim=128（只改容量） ============
log "实验 C：76张 @256, base_dim=128, batch=1, 400轮（从零训）"
rm -rf results_C_severe256_bd128
python -X utf8 train.py --dataset_path "$DS256" --cond_path "$CD256" \
    --output_dir ./results_C_severe256_bd128 \
    --img_size 256 --base_dim 128 --batch_size 1 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 400 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 2>&1 | tail -3
eval_raw results_C_severe256_bd128/models/final_model.pth 256 128 "$CD256" "$DS256" C256

# ============ B：微调 A 域最佳模型（128px） ============
# 注意：--resume 会从 checkpoint 的 epoch 继续（A 模型止于 779），故 --epochs 设 1080 = 再训 300 轮
log "实验 B：微调 A域模型 on 76张 @128, base_dim=128, 再训300轮"
rm -rf results_B_severe128_finetune
python -X utf8 train.py --dataset_path "$DS128" --cond_path "$CD128" \
    --output_dir ./results_B_severe128_finetune \
    --resume results_film_l1lpips/models/final_model.pth \
    --img_size 128 --base_dim 128 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 1080 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 2>&1 | tail -3
eval_raw results_B_severe128_finetune/models/final_model.pth 128 128 "$CD128" "$DS128" B128

log "A / C / B 全部完成"
