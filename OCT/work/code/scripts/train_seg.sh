#!/bin/bash
###############################################################################
# 阶段 4：在 RETOUCH Spectralis 子集上训练下游分割模型（论文的 "downstream model"）
#
# 用法：
#   bash scripts/train_seg.sh [附加参数...]
#
# 默认单折（25% 体积做验证），产物写入 runs/seg/<时间戳>-spectralis/。
#   --folds 4          改为 4 折交叉验证（论文的 Upper 上界模型用 4-fold）
#   --epochs 30        缩短训练（先看趋势时用）
#   --encoder-weights none   不用 ImageNet 预训练（离线；需重新训练后再比较）
#
# 关键约束：
#   1. 训练与评测的 --image-range 必须一致（默认 0-1，见 downstream/seg_lib.py 顶部说明）。
#      两边不一致时 Dice 会静默劣化且不报错，因此本脚本把它显式写死。
#   2. 论文口径：只统计被标注的 B-scan。若 data/retouch/annotated_slices.json 存在
#      会自动传入；否则退化为"跳过掩膜全零切片"并在日志中告警（两种口径不等价，
#      RETOUCH 中"未标注切片"与"已标注但无病灶切片"都表现为全零掩膜）。
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

IMAGE_RANGE="0-1"
CSV="${CSV:-${WORK_ROOT}/data/dataset_configs/retouch_spectralis.csv}"
MANIFEST="${WORK_ROOT}/data/retouch/annotated_slices.json"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_DIR="${OUT_DIR:-${WORK_ROOT}/runs/seg/${STAMP}-spectralis}"

if [ ! -f "${CSV}" ]; then
    echo "❌ 未找到 Spectralis 划分 CSV: ${CSV}" >&2
    echo "   请先运行: bash scripts/build_dataset.sh <RETOUCH 原始目录>" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

MANIFEST_ARG=()
if [ -f "${MANIFEST}" ]; then
    MANIFEST_ARG=(--annotated-slices "${MANIFEST}")
    echo "标注切片清单 : ${MANIFEST}"
else
    echo "⚠️  未找到 ${MANIFEST}，退化为「跳过掩膜全零切片」口径（与论文口径可能有偏差）"
fi

echo "==============================================================="
echo " 训练数据     : ${CSV}（RETOUCH Spectralis）"
echo " 产物目录     : ${OUT_DIR}"
echo " 值域约定     : [${IMAGE_RANGE}]（训练与评测必须一致）"
echo " 编码器       : 见下方参数（默认 ImageNet 预训练）"
echo "==============================================================="

cd "${CODE_DIR}/downstream"
exec "${PY}" train_seg.py \
    --csv "${CSV}" \
    --out-dir "${OUT_DIR}" \
    --image-range "${IMAGE_RANGE}" \
    "${MANIFEST_ARG[@]}" \
    "$@"
