#!/bin/bash
###############################################################################
# 阶段 1：RETOUCH 原始数据 -> 仓库期望的 .npy 布局 + 数据划分 CSV + Hydra 配置
#
# 用法：
#   bash scripts/build_dataset.sh <RETOUCH原始目录> [工作根目录]
#
# 产物：
#   <工作根>/data/retouch/<device>/<NNN>.npy          图像体积，值域 [0, 1]
#   <工作根>/data/retouch/annotations/<device>/<NNN>.npy  多类掩膜，0/1/2/3
#   <工作根>/data/dataset_configs/retouch_<device>.csv
#   <仓库>/preferences/data/retouch_<device>_local.yaml
###############################################################################
set -euo pipefail

RAW_ROOT="${1:?用法: bash scripts/build_dataset.sh <RETOUCH原始目录> [工作根目录]}"
WORK_ROOT="${2:-D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${WORK_ROOT}/code/tta-flow"
PY="F:/python/python.exe"

NPY_ROOT="${WORK_ROOT}/data/retouch"
CSV_DIR="${WORK_ROOT}/data/dataset_configs"
HYDRA_DIR="${REPO_DIR}/preferences/data"

echo "==============================================================="
echo " RETOUCH 原始目录 : ${RAW_ROOT}"
echo " 工作根目录       : ${WORK_ROOT}"
echo " .npy 输出        : ${NPY_ROOT}"
echo " CSV 输出         : ${CSV_DIR}"
echo "==============================================================="

if [ ! -d "${RAW_ROOT}" ]; then
    echo "❌ 原始数据目录不存在: ${RAW_ROOT}" >&2
    exit 2
fi

# 预检：只扫描不写盘，先确认能发现 24/24/22 个体积
echo
echo ">>> 步骤 1/3：预检（不写盘）"
"${PY}" "${CODE_DIR}/preprocess/retouch_to_npy.py" \
    --data-root "${RAW_ROOT}" --out-root "${NPY_ROOT}" --dry-run

echo
echo ">>> 步骤 2/3：转换（预检通过后才会走到这里）"
"${PY}" "${CODE_DIR}/preprocess/retouch_to_npy.py" \
    --data-root "${RAW_ROOT}" --out-root "${NPY_ROOT}"

echo
echo ">>> 步骤 3/3：生成划分 CSV 与 Hydra 配置"
"${PY}" "${CODE_DIR}/preprocess/create_dataset_configs.py" \
    --npy-root "${NPY_ROOT}" \
    --out-dir "${CSV_DIR}" \
    --hydra-dir "${HYDRA_DIR}"

echo
echo "✅ 数据准备完成。生成的 Hydra 配置："
ls -1 "${HYDRA_DIR}"/retouch_*_local.yaml
echo
echo "下一步可运行： bash scripts/train_flow.sh quick"
