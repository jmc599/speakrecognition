#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash finetune_cnceleb_from_wespeaker.sh /path/to/CN-Celeb_flac [target_epochs] [wespeaker_model_path]
#
# Example:
#   bash finetune_cnceleb_from_wespeaker.sh /data/datasets/cn_celeb/CN-Celeb_flac 20 checkpoints/wespeaker/model_5.pt

if [[ $# -lt 1 ]]; then
  echo "Usage: bash finetune_cnceleb_from_wespeaker.sh /path/to/CN-Celeb_flac [target_epochs] [wespeaker_model_path]"
  exit 1
fi

CN_ROOT="$1"
TARGET_EPOCHS="${2:-20}"
WESPK_MODEL="${3:-checkpoints/wespeaker/model_5.pt}"

if [[ ! -d "${CN_ROOT}" ]]; then
  echo "ERROR: CN root not found: ${CN_ROOT}"
  exit 1
fi

if [[ ! -f "lists/train_list_open_set.txt" ]]; then
  echo "ERROR: missing list lists/train_list_open_set.txt"
  exit 1
fi

if [[ ! -f "${WESPK_MODEL}" ]]; then
  echo "ERROR: missing WeSpeaker model: ${WESPK_MODEL}"
  echo "Download example:"
  echo "  mkdir -p checkpoints/wespeaker"
  echo "  wget -O checkpoints/wespeaker/model_5.pt \\"
  echo "    https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM/resolve/main/model_5.pt"
  exit 1
fi

if [[ ! -f "${CN_ROOT}/eval/lists/trials.lst" || ! -f "${CN_ROOT}/eval/lists/enroll.lst" ]]; then
  echo "ERROR: eval lists not found under ${CN_ROOT}/eval/lists"
  exit 1
fi

python train.py \
  --base-path "${CN_ROOT}" \
  --list-path "lists/train_list_open_set.txt" \
  --init-model "${WESPK_MODEL}" \
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
  --metrics-csv "artifacts/results/resnet_v7_wes_init_s1_metrics.csv" \
  --ckpt-prefix "resnet_v7_wes_init_s1"
