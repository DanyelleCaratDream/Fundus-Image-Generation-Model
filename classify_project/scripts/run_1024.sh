#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
echo "########## 1024px  batch=8 (显存限制) ##########"
python -X utf8 classify_project/scripts/train_classifier.py \
  --data fundus/HRSG_4_758_sq1024 --size 1024 \
  --seeds 42 52 62 --epochs 30 --batch-size 8 --num-workers 0 \
  --out classify_project/results/res_1024.json
echo "########## 1024 完成 ##########"
