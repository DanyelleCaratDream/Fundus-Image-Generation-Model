#!/bin/bash
###############################################################################
# quick 档端到端一键流水线：数据 → 训练 → 推理 → 分割训练 → 四行评测
#
# 用法（必须在 nohup 下运行，整条链路约 5–6 小时）：
#   cd D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work
#   mkdir -p logs
#   nohup bash code/scripts/run_quick.sh <RETOUCH原始目录> [cirrus|topcon] \
#       > logs/run_quick.$(date +%Y%m%d-%H%M).log 2>&1 &
#   tail -f logs/run_quick.*.log
#
# 时间预算（RTX 4090 实测外推）：
#   阶段 1 数据转换     数分钟
#   阶段 2 quick 训练   约 2.4 h（30,000 步 @ 约 290 ms/步）
#   阶段 3 推理         约 1.5–2 h（24 卷 × 128 B-scan × 约 2 s/B-scan）
#   阶段 4 分割训练     约 20–40 min
#   阶段 5 四行评测     约 20 min
#
# 断点续跑：每个阶段都先检查产物是否已存在，存在则跳过；因此中断后直接重跑本脚本即可。
###############################################################################
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "用法: bash code/scripts/run_quick.sh <RETOUCH原始目录> [cirrus|topcon]" >&2
    exit 2
fi

RAW_ROOT="$1"
DEVICE="${2:-cirrus}"
WORK_ROOT="${WORK_ROOT:-D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work}"
SCRIPTS="${WORK_ROOT}/code/scripts"
PY="F:/python/python.exe"

START_TS=$(date +%s)
step=0
stage() { step=$((step + 1)); echo; echo "############ 阶段 ${step}: $* ############"; }
elapsed() { echo ">>> 阶段 ${step} 耗时: $(( ($(date +%s) - START_TS) / 60 )) min（累计）"; }

[ -d "${RAW_ROOT}" ] || { echo "❌ RETOUCH 原始目录不存在: ${RAW_ROOT}" >&2; exit 2; }

###############################################################################
stage "环境自检"
###############################################################################
"${PY}" - <<'PYEOF'
import sys, torch
import segmentation_models_pytorch as smp
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | smp {smp.__version__}")
if not torch.cuda.is_available():
    sys.exit("❌ CUDA 不可用")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | 显存 {p.total_memory / 1024**3:.1f} GiB | 计算能力 {p.major}.{p.minor}")
PYEOF
"${PY}" "${WORK_ROOT}/code/tests/check_source_integrity.py" \
    --root "${WORK_ROOT}/code" --quiet || {
    echo "❌ 源码完整性检查未通过，请先修复再执行实验" >&2; exit 3; }
echo "✅ 环境自检通过"

###############################################################################
stage "数据转换（.mhd/.raw -> .npy + 划分 CSV + Hydra 配置）"
###############################################################################
if [ -d "${WORK_ROOT}/data/retouch/spectralis" ] && \
   [ -f "${WORK_ROOT}/data/dataset_configs/retouch_${DEVICE}.csv" ]; then
    echo "⏭  已存在转换产物，跳过（如需重转请删除 ${WORK_ROOT}/data/retouch 后重跑）"
else
    bash "${SCRIPTS}/build_dataset.sh" "${RAW_ROOT}" "${WORK_ROOT}"
fi
elapsed

###############################################################################
stage "训练 Flow Matching 生成模型（quick 档：30,000 步）"
###############################################################################
EXP_DIR="$(ls -1dt "${WORK_ROOT}"/runs/*-quick30k 2>/dev/null | head -1 || true)"
if [ -n "${EXP_DIR}" ] && ls -1 "${EXP_DIR}"/checkpoints/*.ckpt >/dev/null 2>&1; then
    echo "⏭  已存在训练产物，跳过: ${EXP_DIR}"
else
    bash "${SCRIPTS}/train_flow.sh" quick
    EXP_DIR="$(ls -1dt "${WORK_ROOT}"/runs/*-quick30k | head -1)"
fi
echo "实验目录: ${EXP_DIR}"
elapsed

###############################################################################
stage "跨设备推理（${DEVICE} -> 重建为 Spectralis 风格）"
###############################################################################
if ls -1 "${EXP_DIR}"/inference/outputs/pred_*.npy >/dev/null 2>&1; then
    echo "⏭  已存在推理产物，跳过（如需重跑请删除 ${EXP_DIR}/inference/outputs）"
else
    bash "${SCRIPTS}/infer_flow.sh" "${EXP_DIR}" "${DEVICE}"
fi
elapsed

###############################################################################
stage "训练下游分割模型（RETOUCH Spectralis 子集）"
###############################################################################
SEG_CKPT="$(ls -1t "${WORK_ROOT}"/runs/seg/*/fold0/best.pt 2>/dev/null | head -1 || true)"
if [ -n "${SEG_CKPT}" ]; then
    echo "⏭  已存在分割模型，跳过: ${SEG_CKPT}"
else
    bash "${SCRIPTS}/train_seg.sh"
    SEG_CKPT="$(ls -1t "${WORK_ROOT}"/runs/seg/*/fold0/best.pt | head -1)"
fi
echo "分割模型: ${SEG_CKPT}"
elapsed

###############################################################################
stage "四行评测（Baseline / TENT / CoTTA / Ours）+ 汇总表"
###############################################################################
bash "${SCRIPTS}/eval_all.sh" "${EXP_DIR}" "${DEVICE}" "${SEG_CKPT}"
elapsed

echo
echo "############ 全部完成 ############"
echo "总耗时: $(( ($(date +%s) - START_TS) / 60 )) min"
echo "实验目录: ${EXP_DIR}"
echo "汇总表  : ${EXP_DIR}/eval/summary.md"
echo
cat "${EXP_DIR}/eval/summary.md"
