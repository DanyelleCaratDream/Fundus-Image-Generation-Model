#!/usr/bin/env bash
# 夜间任务：在老师数据集的 76 张 train/severe 上，从零训一个条件 DDPM @ 256。
#
# 为什么用这套配置：
#   · 数据只能用 train/severe 的 76 张 —— 扩充后的数据集要训分类器，所以不能碰 val/test
#   · 256×256 —— 由分辨率消融实验定的（512 显存扛不住，256 是本机折中）
#   · base_dim=64, batch=2 —— 256 下实测 4.56 GB，留足余量；76 张数据也不宜用太大模型
#   · 条件生成（血管 mask）—— 比无条件容易得多，76 张才可能训得动
#   · L1 + LPIPS + FiLM + EMA 0.9999 —— 复现最佳模型的配方
#
# 脚本会先等 GPU 空出来（分辨率消融还在跑），再开始训练。
set -u

cd "$(dirname "$0")"

OUT=./results_severe256
LOG=/tmp/overnight_severe256.log

echo "=============================================================="
echo " 夜间任务：severe 76 张 @ 256 条件 DDPM"
echo " 输出: $OUT"
echo " 开始: $(date '+%F %T')"
echo "=============================================================="

# ---- 1. 等 GPU 空出来 ----
echo "[1/3] 等 GPU 空闲 ..."
waited=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt 1500 ]; then
        echo "      GPU 已空闲（占用 ${used} MiB），等待 ${waited}s"
        break
    fi
    sleep 60
    waited=$((waited + 60))
    if [ $((waited % 300)) -eq 0 ]; then
        echo "      已等待 ${waited}s，当前占用 ${used:-?} MiB"
    fi
    if [ "$waited" -gt 7200 ]; then
        echo "      等超过 2 小时，放弃"
        exit 1
    fi
done

# ---- 2. 试跑 3 轮，测速度 + 验证配置 ----
echo "[2/3] 试跑 3 轮验证配置 ..."
python -X utf8 train.py \
    --dataset_path ../../../../fundus/severe_train_sq256 \
    --cond_path ./conditions_severe_train \
    --output_dir /tmp/_smoke_severe256 \
    --img_size 256 --base_dim 64 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 3 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --model_save_interval 100 --image_save_interval 100 \
    --num_workers 0 > "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "      试跑失败（退出码 $rc），日志尾部："
    tail -20 "$LOG" | sed 's/^/      /'
    exit 1
fi
echo "      试跑通过"
rm -rf /tmp/_smoke_severe256

# ---- 3. 正式训练 ----
echo "[3/3] 正式训练开始 $(date '+%F %T')"
rm -rf "$OUT"
python -X utf8 train.py \
    --dataset_path ../../../../fundus/severe_train_sq256 \
    --cond_path ./conditions_severe_train \
    --output_dir "$OUT" \
    --img_size 256 --base_dim 64 --batch_size 2 \
    --dim_mults 1 2 3 4 --attn_layers 2 \
    --epochs 800 --use_lpips --lpips_weight 0.1 --color_weight 0.01 \
    --ema_decay 0.9999 --use_scale_shift_norm --cool_sleep -0.5 \
    --model_save_interval 50 --image_save_interval 25 \
    --preview_grid_size 4 --num_workers 0 --seed 42 \
    > "$LOG" 2>&1
echo "     训练结束 $(date '+%F %T')，退出码 $?"
echo "=============================================================="
