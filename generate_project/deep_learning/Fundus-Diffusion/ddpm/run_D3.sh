#!/usr/bin/env bash
# DL-EX-021（D3）：A 域最佳微调 + 干净 193 张 @128（2026-10-17）
# 与 DL-EX-018 只差数据量（76→193 唯一，×2.9）的单变量对照。
# 预注册/parity 表：generate_project/deep_learning/EXPERIMENTS.md → DL-EX-021
# 判定：只看 --use_raw 出图 vs 018 的 2/8；256 交付走 SR 管线（报告 5.11.2），不冒充原生 256
set -u
cd "$(dirname "$0")"

DS=../../../../fundus/combined_sq128
CD=./conditions_combined_128

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

echo "  数据: $DS（$(ls $DS | wc -l) 张）"
echo "  条件: $CD（$(ls $CD | wc -l) 张）"
wait_gpu || { echo "  GPU 等待超时"; exit 1; }

log "D3：微调 film_l1lpips(779) → combined 193 @128 bd128，epochs 1100（+321，同 018）"
rm -rf results_D3_combined128_finetune
python -X utf8 train.py --dataset_path "$DS" --cond_path "$CD" \
    --output_dir ./results_D3_combined128_finetune \
    --img_size 128 --base_dim 128 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --resume ./results_film_l1lpips/models/final_model.pth \
    --epochs 1100 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 > /tmp/D3.log 2>&1
log "D3 训练结束"
rm -rf /tmp/evalraw_D3
python -X utf8 generate.py --checkpoint results_D3_combined128_finetune/models/final_model.pth \
    --num_images 8 --output_dir /tmp/evalraw_D3 --grid_size 0 \
    --img_size 128 --base_dim 128 --dim_mults 1 2 3 4 --attn_layers 2 \
    --sampler ddim --sampling_steps 100 --use_raw --cond_path "$CD" > /dev/null 2>&1
echo "  → /tmp/evalraw_D3/singles（$(ls /tmp/evalraw_D3/singles 2>/dev/null | wc -l) 张）"
log "D3 完成"
