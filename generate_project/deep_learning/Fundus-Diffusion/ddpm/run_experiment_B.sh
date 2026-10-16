#!/usr/bin/env bash
# 实验 B：微调 A 域最佳模型，使其适配老师数据集的 severe 类。
#
# 背景：
#   · DL-EX-015（76张@256, base_dim=64）失败 —— 生成糊块
#   · DL-EX-016（76张@128, base_dim=64）同样失败 —— 排除「分辨率」假说
#   · 两个失败实验的 loss 都正常收敛（0.05~0.08），说明模型能拟合训练目标，
#     失败在「没能形成数据分布的全局先验」→ 倾向是数据量问题（76 张太少）
#
# 思路：不从头学「眼底图长什么样」，而是把这个知识从 A 域模型迁移过来，
#       只用 76 张把它「调」到老师数据的风格上。
#
# 注意：--resume 会从 checkpoint 记录的 epoch 继续（A 模型止于 779），
#       故 --epochs 设 1080 = 再训 300 轮。
set -u
cd "$(dirname "$0")"

DS=../../../../fundus/severe_train_sq128
CD=./conditions_severe_train_128
BASE=results_film_l1lpips/models/final_model.pth
OUT=results_B_severe128_finetune

log() { echo; echo "=============================================================="; \
        echo "[$(date '+%F %T')] $*"; echo "=============================================================="; }

wait_gpu() {
    local w=0
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        [ -n "$used" ] && [ "$used" -lt 1500 ] && { echo "  GPU 已空闲（${used} MiB）"; return 0; }
        sleep 30; w=$((w+30))
        [ $((w % 300)) -eq 0 ] && echo "  等待 GPU 空闲 ${w}s"
        [ "$w" -gt 1200 ] && return 1
    done
}

log "实验 B：微调 A 域最佳模型 on severe 76 张 @128（再训 300 轮）"
echo "  基础模型: $BASE"
echo "  数据    : $DS（76 张）"
echo "  条件    : $CD（76 张血管 mask）"
echo

wait_gpu || exit 1

rm -rf "$OUT"
python -X utf8 train.py \
    --dataset_path "$DS" --cond_path "$CD" --output_dir "./$OUT" \
    --resume "$BASE" \
    --img_size 128 --base_dim 128 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 1080 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 2>&1 | tail -5

log "B 训练结束，用原始权重出图评估"
rm -rf /tmp/evalraw_B
python -X utf8 generate.py --checkpoint "$OUT/models/final_model.pth" \
    --num_images 8 --output_dir /tmp/evalraw_B --grid_size 0 \
    --img_size 128 --base_dim 128 --dim_mults 1 2 3 4 --attn_layers 2 \
    --sampler ddim --sampling_steps 100 --use_raw \
    --cond_path "$CD" --dataset_path "$DS" > /dev/null 2>&1
echo "  → /tmp/evalraw_B/singles（$(ls /tmp/evalraw_B/singles 2>/dev/null | wc -l) 张）"

log "实验 B 完成"
