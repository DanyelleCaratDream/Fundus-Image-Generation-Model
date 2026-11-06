#!/bin/bash
###############################################################################
# 阶段 3：用训练好的 Flow Matching 模型做跨设备推理
#
# 用法：
#   bash scripts/infer_flow.sh <实验目录> <目标设备> [附加 Hydra 覆盖...]
#
# 例：
#   bash scripts/infer_flow.sh runs/2026-10-25_03-00-00-quick30k cirrus
#
# 常用覆盖：
#   test_parameters.num_reference_trajectories=0      关闭直方图匹配（= 论文 FM [14] 基线）
#   test_parameters.s_target=0.75                     s_target 敏感性扫描
#   test_parameters.ode_start=from_s_target           改从 s_target 起步积分（论文文字描述）
#
# 说明：
#   - 目标设备（cirrus/topcon）的图像即"测试时"输入；Spectralis 是模型训练域
#   - 输出写入 <实验目录>/inference/outputs/pred_%03d.npy，值域 [0,1]
#   - 架构参数必须与训练一致，故复用 _common.sh 中的 PAPER_ARCH
###############################################################################
set -euo pipefail

EXP_DIR="${1:?用法: bash scripts/infer_flow.sh <实验目录> <cirrus|topcon> [覆盖...]}"
DEVICE="${2:?请指定目标设备: cirrus 或 topcon}"
shift 2 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

case "${DEVICE}" in
  cirrus)     DATA_CFG="retouch_cirrus_local";;
  topcon)     DATA_CFG="retouch_topcon_local";;
  spectralis) DATA_CFG="retouch_spectralis_local";;
  *) echo "❌ 未知设备: ${DEVICE}（可选 cirrus|topcon|spectralis）" >&2; exit 2;;
esac

if [ ! -d "${EXP_DIR}" ]; then
    echo "❌ 实验目录不存在: ${EXP_DIR}" >&2
    exit 2
fi

cd "${REPO_DIR}"

OUT_DIR="${EXP_DIR}/inference/outputs"
# 推理每卷约 2.2 s/B-scan；Cirrus 24 卷 x 128 B-scan 约 1.9 h
echo "==============================================================="
echo " 实验目录   : ${EXP_DIR}"
echo " 目标设备   : ${DEVICE}  (${DATA_CFG})"
arch_desc
echo " 参考轨迹数 : ${NREF:-仓库默认 50}"
echo " s_target   : ${STARGET:-仓库默认 0.8}"
echo " ODE 起点   : ${ODESTART:-仓库默认 from_zero}"
echo " 输出目录   : ${OUT_DIR}"
echo "==============================================================="

# 清理上一次推理的残留，避免 pred_*.npy 与 CSV 行序错配
rm -rf "${OUT_DIR}"

"${PY}" src/test.py \
    +experiment_path="${EXP_DIR}" \
    data="${DATA_CFG}" \
    hydra.run.dir="${EXP_DIR}/inference" \
    "${PAPER_ARCH[@]}" \
    "${PAPER_RESIZE_TEST[@]}" \
    "$@"

echo
echo "✅ 推理完成"
echo -n "   输出文件数（应为该设备的 volume 数）："
ls -1 "${OUT_DIR}"/pred_*.npy 2>/dev/null | wc -l
echo
echo "下一步：用评测脚本算 Dice（--source-range 为 0-1，与仓库约定一致）"
