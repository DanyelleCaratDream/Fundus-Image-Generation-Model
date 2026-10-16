#!/usr/bin/env bash
# 一键查看 A/C/B 实验链的进度
# 用法：bash generate_project/deep_learning/Fundus-Diffusion/ddpm/check_progress.sh
cd "$(dirname "$0")" || exit 1

declare -A EXP=(
  ["A · sev76@128 base_dim64 (隔离分辨率)"]="results_A_severe128_scratch:800"
  ["C · sev76@256 base_dim128 (隔离容量)"]="results_C_severe256_bd128:400"
  ["B · 微调A域模型@128       (迁移学习)"]="results_B_severe128_finetune:300"
)
ORDER=(
  "A · sev76@128 base_dim64 (隔离分辨率)"
  "C · sev76@256 base_dim128 (隔离容量)"
  "B · 微调A域模型@128       (迁移学习)"
)

echo "════════════════════════════════════════════════════════════════"
echo "  实验链进度   $(date '+%F %T')"
echo "════════════════════════════════════════════════════════════════"

running=""
for name in "${ORDER[@]}"; do
    dir="${EXP[$name]%%:*}"; total="${EXP[$name]##*:}"
    log="$dir/logs/training.log"
    if [ ! -d "$dir" ]; then
        printf "  %-38s  未开始\n" "$name"
        continue
    fi
    if [ ! -f "$log" ]; then
        printf "  %-38s  已建目录，日志未生成\n" "$name"
        continue
    fi
    done_n=$(grep -c '^Epoch' "$log" 2>/dev/null || echo 0)
    last=$(grep '^Epoch' "$log" 2>/dev/null | tail -1)
    loss=$(echo "$last" | sed -n 's/.*loss=\([0-9.]*\).*/\1/p')
    tm=$(echo "$last" | sed -n 's/.*time=\([0-9.]*\)s.*/\1/p')
    # 近期中位数（最后 20 轮）——用最后一轮算会被 checkpoint/预览的尖峰带偏
    med=$(grep '^Epoch' "$log" 2>/dev/null | tail -20 \
          | sed -n 's/.*time=\([0-9.]*\)s.*/\1/p' | sort -n \
          | awk '{a[NR]=$1} END{if(NR==0){print 0}else{print a[int((NR+1)/2)]}}')
    if [ "$done_n" -ge "$total" ]; then
        pct=100; mark="✅ 完成"
    else
        pct=$(( done_n * 100 / total )); mark="🏃 进行中"
        running="$name"
    fi
    printf "  %-38s  %3d/%-3d (%3d%%)  loss=%-9s 中位%s s/轮  %s\n" \
           "$name" "$done_n" "$total" "$pct" "$loss" "$med" "$mark"
    if [ -n "$running" ] && [ -n "$med" ] && [ "$med" != "0" ]; then
        rest=$(( total - done_n ))
        eta=$(python -c "print(f'{$rest*float(\"$med\")/60:.0f}')" 2>/dev/null)
        [ -n "$eta" ] && printf "  %-38s  → 预计还需 %s 分钟\n" "" "$eta"
    fi
done

echo "────────────────────────────────────────────────────────────────"
printf "  GPU: "
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null \
    | sed 's/, / | /g'
echo "════════════════════════════════════════════════════════════════"
