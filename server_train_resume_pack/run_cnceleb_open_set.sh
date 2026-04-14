#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_cnceleb_open_set.sh /path/to/CN-Celeb_flac

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_cnceleb_open_set.sh /path/to/CN-Celeb_flac"
  exit 1
fi

CN_ROOT="$1"

python make_open_set_list.py \
  --dataset-root "${CN_ROOT}" \
  --include-subdir data \
  --eval-trials "${CN_ROOT}/eval/lists/trials.lst" \
  --eval-enroll "${CN_ROOT}/eval/lists/enroll.lst" \
  --output-list "lists/train_list_open_set.txt"

python train.py \
  --base-path "${CN_ROOT}" \
  --list-path "lists/train_list_open_set.txt" \
  --epochs 20 \
  --batch-size 32 \
  --max-frames 300 \
  --num-workers 6 \
  --monitor-base-path "${CN_ROOT}/eval" \
  --monitor-trials "${CN_ROOT}/eval/lists/trials.lst" \
  --monitor-enroll-list "${CN_ROOT}/eval/lists/enroll.lst" \
  --monitor-limit 2000 \
  --monitor-sample-mode balanced \
  --preprocess-for-inference \
  --show-grad-norm \
  --metrics-csv "artifacts/results/open_set_metrics.csv" \
  --ckpt-prefix "open_set_run"
