#!/usr/bin/env bash
# 分辨率消融：128 / 256 / 512，各 3 个种子
set -u
cd "$(dirname "$0")/../.."   # 回到项目根
mkdir -p classify_project/results
for S in 128 256 512; do
  echo "########## 分辨率 ${S} ##########"
  python -X utf8 classify_project/scripts/train_classifier.py \
    --data "fundus/HRSG_4_758_sq${S}" --size "${S}" \
    --seeds 42 52 62 --epochs 30 --num-workers 0 \
    --out "classify_project/results/res_${S}.json"
done
echo "########## 全部完成 ##########"
