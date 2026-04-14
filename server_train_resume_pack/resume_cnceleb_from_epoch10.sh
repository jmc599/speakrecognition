#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash resume_cnceleb_from_epoch10.sh /path/to/CN-Celeb_flac [target_epochs]
#
# Example:
#   bash resume_cnceleb_from_epoch10.sh /data/datasets/cn_celeb/CN-Celeb_flac 20

if [[ $# -lt 1 ]]; then
  echo "Usage: bash resume_cnceleb_from_epoch10.sh /path/to/CN-Celeb_flac [target_epochs]"
  exit 1
fi

CN_ROOT="$1"
TARGET_EPOCHS="${2:-20}"

if [[ ! -d "${CN_ROOT}" ]]; then
  echo "ERROR: CN root not found: ${CN_ROOT}"
  exit 1
fi

if [[ ! -f "checkpoints/resnet_v7_open_s1_latest.pth" ]]; then
  echo "ERROR: missing checkpoint checkpoints/resnet_v7_open_s1_latest.pth"
  exit 1
fi

if [[ ! -f "lists/train_list_open_set.txt" ]]; then
  echo "ERROR: missing list lists/train_list_open_set.txt"
  exit 1
fi

python train.py \
  --base-path "${CN_ROOT}" \
  --list-path "lists/train_list_open_set.txt" \
  --resume "checkpoints/resnet_v7_open_s1_latest.pth" \
  --epochs "${TARGET_EPOCHS}" \
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
  --metrics-csv "artifacts/results/resnet_v7_open_s1_metrics.csv" \
  --ckpt-prefix "resnet_v7_open_s1"
