#!/usr/bin/env bash
set -euo pipefail

# Experiment: balanced_speaker + augmentation + more epochs, from scratch
#
# Changes vs open_s2 (epochs=30, lr=1e-3, CosineAnnealingLR, no_augment=True):
#   1. --train-sampler balanced_speaker  (was random)
#   2. augmentation ON                   (was --no-augment)
#   3. --epochs 40                       (was 30)
#   4. --train-sample-repeat-cap 3       (new, controls oversampling of rare speakers)
#
# Usage:
#   bash run_balanced_augment_scratch.sh /path/to/CN-Celeb_flac

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_balanced_augment_scratch.sh /path/to/CN-Celeb_flac"
  exit 1
fi

CN_ROOT="$1"

if [[ ! -d "${CN_ROOT}" ]]; then
  echo "ERROR: CN root not found: ${CN_ROOT}"
  exit 1
fi

if [[ ! -f "lists/train_list_open_set.txt" ]]; then
  echo "ERROR: missing list lists/train_list_open_set.txt"
  exit 1
fi

if [[ ! -f "${CN_ROOT}/eval/lists/trials.lst" || ! -f "${CN_ROOT}/eval/lists/enroll.lst" ]]; then
  echo "ERROR: eval lists not found under ${CN_ROOT}/eval/lists"
  exit 1
fi

python train.py \
  --base-path "${CN_ROOT}" \
  --list-path "lists/train_list_open_set.txt" \
  --train-sampler balanced_speaker \
  --train-sample-repeat-cap 3 \
  --epochs 40 \
  --batch-size 32 \
  --lr 1e-3 \
  --min-lr 1e-6 \
  --aam-scale 30 \
  --aam-margin 0.20 \
  --max-frames 300 \
  --num-workers 6 \
  --monitor-base-path "${CN_ROOT}/eval" \
  --monitor-trials "${CN_ROOT}/eval/lists/trials.lst" \
  --monitor-enroll-list "${CN_ROOT}/eval/lists/enroll.lst" \
  --monitor-limit 2000 \
  --monitor-sample-mode balanced \
  --preprocess-for-inference \
  --show-grad-norm \
  --metrics-csv "artifacts/results/resnet_v7_balanced_aug_s3_metrics.csv" \
  --ckpt-prefix "resnet_v7_balanced_aug_s3"
