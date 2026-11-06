#!/bin/bash
###############################################################################
# 阶段 5：对某个实验目录 + 目标设备，一次性产出与论文 Table 1 同构的四行对照
#
# 用法：
#   bash scripts/eval_all.sh <实验目录> <cirrus|topcon> [分割模型 ckpt]
#
# 产出（写入 <实验目录>/eval/）：
#   baseline.json   分割模型直接推理目标设备原始图像（无任何适配）
#   ours.json       分割模型推理 flow matching 重建结果（Ours_uncond. 行）
#   tent.json       TENT 在线适配（无标签目标域上的熵最小化）
#   cotta.json      CoTTA 在线适配（伪标签 + 增强一致性 + EMA 教师）
#   crosscheck.json 两条独立实现路径（eval_dice.py 与 eval_tta_seg.py --mode none）
#                   的一致性检查——自测中已验证应逐位相同，此处作为运行时守卫
#   summary.md      对照表，可直接与论文数值比较
#
# 对照设计（必须遵守，否则差值不可归因）：
#   * 四行共用同一份分割权重、同一份设备划分 CSV、同一批被标注切片
#   * TENT / CoTTA 作用于**分割模型本身**（在线更新其权重），不产生重建图像，
#     因此它们的输入是目标设备**原始图像**；论文表格中这两行 FID 为 N/A 即是佐证
#   * Ours 一行与 Baseline 的唯一差别是"输入换成了重建图像"
###############################################################################
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "用法: bash scripts/eval_all.sh <实验目录> <cirrus|topcon> [分割模型ckpt]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP_DIR="$(cd "$1" && pwd)"
DEVICE="$2"
CKPT="${3:-}"

IMAGE_RANGE="0-1"          # 必须与 train_seg.sh / seg_lib.py 一致
EVAL_BATCH="${EVAL_BATCH:-16}"   # TENT/CoTTA 依赖批统计量，须 >1
MANIFEST_JSON="${WORK_ROOT}/data/retouch/annotated_slices.json"
RECON_DIR="${EXP_DIR}/inference/outputs"
OUT_DIR="${EXP_DIR}/eval"

case "${DEVICE}" in
  cirrus) DATA_CFG="retouch_cirrus";;
  topcon) DATA_CFG="retouch_topcon";;
  *) echo "❌ 未知设备: ${DEVICE}（本脚本只处理 cirrus|topcon 的跨设备方向）" >&2; exit 2;;
esac
CSV="${WORK_ROOT}/data/dataset_configs/${DATA_CFG}.csv"
RAW_DIR="${WORK_ROOT}/data/retouch/${DEVICE}"

# ---------------------------------------------------------------------------
# 前置检查：任何一项不满足都直接退出，不做"尽力而为"的降级
# ---------------------------------------------------------------------------
FAIL=0
[ -f "${CSV}" ]       || { echo "❌ 缺少划分 CSV: ${CSV}" >&2; FAIL=1; }
[ -d "${RAW_DIR}" ]   || { echo "❌ 缺少原始设备图像目录: ${RAW_DIR}" >&2; FAIL=1; }
[ -d "${RECON_DIR}" ] || { echo "❌ 缺少重建输出目录: ${RECON_DIR}（先跑 infer_flow.sh）" >&2; FAIL=1; }
if [ -z "${CKPT}" ]; then
    CKPT="$(ls -1t "${WORK_ROOT}"/runs/seg/*/fold0/best.pt 2>/dev/null | head -1 || true)"
    [ -n "${CKPT}" ] || { echo "❌ 未找到分割模型 checkpoint，请先跑 train_seg.sh 或显式传入第 3 个参数" >&2; FAIL=1; }
fi
[ "${FAIL}" -eq 0 ] || exit 2

N_PRED="$(ls -1 "${RECON_DIR}"/pred_*.npy 2>/dev/null | wc -l)"
N_ROW="$(($(wc -l < "${CSV}") - 1))"
if [ "${N_PRED}" != "${N_ROW}" ]; then
    echo "❌ 重建文件数(${N_PRED}) 与 CSV 行数(${N_ROW}) 不一致，序号会错配。" >&2
    echo "   请确认推理时使用的 CSV 与本次评测的 ${DATA_CFG}.csv 相同，并重跑 infer_flow.sh" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

MANIFEST_ARG=()
if [ -f "${MANIFEST_JSON}" ]; then
    MANIFEST_ARG=(--slice-manifest "${MANIFEST_JSON}")
    echo "标注切片清单 : ${MANIFEST_JSON}"
else
    echo "⚠️  未找到 ${MANIFEST_JSON}：退化为「跳过掩膜全零切片」口径。"
    echo "   RETOUCH 中『未标注切片』与『已标注但无病灶切片』都表现为全零掩膜，"
    echo "   两种口径不等价；若与论文数值出现系统性偏差，请优先排查此处。"
fi

echo "==============================================================="
echo " 实验目录     : ${EXP_DIR}"
echo " 目标设备     : ${DEVICE}"
echo " 分割模型     : ${CKPT}"
echo " 重建输出     : ${RECON_DIR}（${N_PRED} 个文件）"
echo " 评测批大小   : ${EVAL_BATCH}"
echo "==============================================================="
echo

run_eval_dice() {
    local mode="$1"; shift
    echo ">>> [${mode}] eval_dice.py $*"
    "${PY}" "${CODE_DIR}/eval/eval_dice.py" \
        --csv "${CSV}" --ckpt "${CKPT}" \
        --image-range "${IMAGE_RANGE}" --source-range "0-1" \
        --batch-size "${EVAL_BATCH}" "${MANIFEST_ARG[@]}" "$@"
    echo
}

# ---------------------------------------------------------------------------
# 1) Baseline：分割模型直接推理目标设备原始图像
# 2) Ours    ：输入换成 flow matching 重建结果，其余完全相同
# ---------------------------------------------------------------------------
run_eval_dice baseline --raw-dir "${RAW_DIR}"   --out "${OUT_DIR}/baseline.json"
run_eval_dice ours     --recon-dir "${RECON_DIR}" --out "${OUT_DIR}/ours.json"

# ---------------------------------------------------------------------------
# 3) 交叉校验：eval_tta_seg.py --mode none 应与 baseline 逐位一致。
#    这是两条独立代码路径（seg_lib 数据集 vs eval_dice 的逐体积循环）互证，
#    自测中已确认逐位相同；此处若出现差异，说明两次运行的数据处理有分歧，
#    后续 TENT/CoTTA 的差值都不可信，必须排查。
# ---------------------------------------------------------------------------
echo ">>> [crosscheck] eval_tta_seg.py --mode none"
"${PY}" "${CODE_DIR}/baselines/eval_tta_seg.py" \
    --csv "${CSV}" --ckpt "${CKPT}" --mode none \
    --out "${OUT_DIR}/crosscheck.json" \
    --batch-size "${EVAL_BATCH}" --image-range "${IMAGE_RANGE}" \
    "${MANIFEST_ARG[@]}"
echo

# ---------------------------------------------------------------------------
# 4) TENT / CoTTA：在目标设备原始图像上做在线适配
#    注意：这一步的差值**同时包含两部分**——(a) 权重适配本身，(b) 归一化统计量
#    从源域 running stats 切换到测试批统计量（TENT/CoTTA 的必要条件）。
#    Baseline 使用源域 stats，因此差值不能全部归因于适配策略，报告中须如实披露。
#    --tent-source-stats 可拆出 (b) 的贡献作为对照（已声明为偏离原设定）。
# ---------------------------------------------------------------------------
for MODE in tent cotta; do
    echo ">>> [${MODE}] eval_tta_seg.py --mode ${MODE}"
    "${PY}" "${CODE_DIR}/baselines/eval_tta_seg.py" \
        --csv "${CSV}" --ckpt "${CKPT}" --mode "${MODE}" \
        --out "${OUT_DIR}/${MODE}.json" \
        --batch-size "${EVAL_BATCH}" --image-range "${IMAGE_RANGE}" \
        "${MANIFEST_ARG[@]}"
    echo
done

# ---------------------------------------------------------------------------
# 5) 汇总为 markdown 对照表（与论文 Table 1 同构）
# ---------------------------------------------------------------------------
"${PY}" "${CODE_DIR}/eval/summarize.py" --eval-dir "${OUT_DIR}" --device "${DEVICE}"

echo
echo "✅ 评测完成，汇总表: ${OUT_DIR}/summary.md"
