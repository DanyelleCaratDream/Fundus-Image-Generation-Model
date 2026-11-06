#!/bin/bash
###############################################################################
# 阶段 2：Flow Matching 生成模型训练
#
# 用法：
#   bash scripts/train_flow.sh <档位> [附加 Hydra 覆盖...]
#
# 档位：
#   smoke   501 步，仅验证链路能跑通（约 3 分钟）
#   quick   30,000 步（用户选定的 quick 档）
#   full    300,000 步（论文设定）
#
# 实测速度（RTX 4090，224x224，约 108M 参数，batch 8 + 累积 2）：
#   约 290 ms/次优化器步进 -> quick 档约 2.4 h，full 档约 24 h
###############################################################################
set -euo pipefail

TIER="${1:?用法: bash scripts/train_flow.sh <smoke|quick|full> [覆盖...]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

DATA_CFG="${DATA_CFG:-retouch_spectralis_local}"

case "${TIER}" in
  smoke) ITERS=501;    EVAL_STEPS=250;  COMMENT="smoke501";;
  quick) ITERS=30001;  EVAL_STEPS=1500; COMMENT="quick30k";;
  full)  ITERS=300001; EVAL_STEPS=2500; COMMENT="full300k";;
  *) echo "❌ 未知档位: ${TIER}（可选 smoke|quick|full）" >&2; exit 2;;
esac

# 允许环境变量覆盖迭代数（便于断点续训或缩短验证）
ITERS="${ITERS_OVERRIDE:-${ITERS}}"

cd "${REPO_DIR}"

echo "==============================================================="
echo " 档位       : ${TIER}（迭代 ${ITERS} 次优化器步进）"
arch_desc
echo " 数据配置   : ${DATA_CFG}"
echo " 评估步长   : 每 ${EVAL_STEPS} 步存 checkpoint 并出验证样本"
echo " 产物目录   : ${REPO_DIR}/runs/<时间戳>-${COMMENT}/"
echo "==============================================================="

exec "${PY}" src/train.py \
    data="${DATA_CFG}" \
    "${PAPER_ARCH[@]}" \
    "${PAPER_RESIZE_TRAIN[@]}" \
    train_parameters.max_iterations="${ITERS}" \
    train_parameters.eval_num_steps="${EVAL_STEPS}" \
    train_parameters.train_batch_size="${BATCH}" \
    train_parameters.grad_accum_steps="${GRAD_ACCUM}" \
    train_parameters.comment="${COMMENT}" \
    "$@"
