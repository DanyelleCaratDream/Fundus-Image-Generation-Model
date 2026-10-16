#!/usr/bin/env bash
# 把实验 B（微调 A 域模型）从 epoch 850 续跑到 1100。
#
# 背景：B 原本要训 300 轮（780→1079），但在 epoch 878 被中断。
#      检查点只到 850，所以从 850 续跑（丢掉 28 轮，影响不大）。
#
# ⚠️ 本脚本不清空输出目录——那里有 checkpoint_epoch_000800/000850，清掉就没了。
set -u
cd "$(dirname "$0")"

DS=../../../../fundus/severe_train_sq128
CD=./conditions_severe_train_128
CKPT=results_B_severe128_finetune/models/checkpoint_epoch_000850.pth
OUT=results_B_severe128_finetune
LOG=/tmp/B_continue.log

log() { echo; echo "=============================================================="; \
        echo "[$(date '+%F %T')] $*"; echo "=============================================================="; }

log "接续实验 B：从 epoch 850 训到 1100（250 轮）"
echo "  数据   : $DS（$(ls $DS 2>/dev/null | wc -l) 张）"
echo "  条件   : $CD（$(ls $CD 2>/dev/null | wc -l) 张）"
echo "  续训起点: $CKPT"
echo "  输出   : $OUT（不清空，保留已有检查点）"
echo

if [ ! -f "$CKPT" ]; then echo "  ❌ 找不到检查点"; exit 1; fi
if [ "$(ls $DS | wc -l)" -lt 10 ]; then echo "  ❌ 训练数据为空"; exit 1; fi

# 等 GPU
w=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 1500 ] && { echo "  GPU 空闲（${used} MiB）"; break; }
    sleep 30; w=$((w+30))
    [ $((w % 300)) -eq 0 ] && echo "  等待 GPU 空闲 ${w}s"
    [ "$w" -gt 900 ] && { echo "  等待超时"; exit 1; }
done

python -X utf8 train.py \
    --dataset_path "$DS" --cond_path "$CD" --output_dir "./$OUT" \
    --resume "$CKPT" \
    --img_size 128 --base_dim 128 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 1100 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 > "$LOG" 2>&1

log "B 续训结束，用原始权重出 8 张图"
rm -rf /tmp/evalraw_B2
python -X utf8 generate.py --checkpoint "$OUT/models/final_model.pth" \
    --num_images 8 --output_dir /tmp/evalraw_B2 --grid_size 0 \
    --img_size 128 --base_dim 128 --dim_mults 1 2 3 4 --attn_layers 2 \
    --sampler ddim --sampling_steps 100 --use_raw \
    --cond_path "$CD" > /dev/null 2>&1
echo "  → /tmp/evalraw_B2/singles（$(ls /tmp/evalraw_B2/singles 2>/dev/null | wc -l) 张）"

log "完成：B 已训到 1100 轮"
