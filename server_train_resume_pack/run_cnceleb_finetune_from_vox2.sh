#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_cnceleb_finetune_from_vox2.sh /path/to/CN-Celeb_flac [vox2_pretrain_checkpoint] [target_epochs]
#
# Example:
#   bash run_cnceleb_finetune_from_vox2.sh \
#     /data/datasets/cn_celeb/cn-celeb_v2/CN-Celeb_flac \
#     checkpoints/resnet_v7_vox2_aac12_pt_best.pth \
#     20
#
# Notes:
#   - This stage loads the Stage 1 checkpoint with --init-model.
#   - The classification head is rebuilt for CN-Celeb open-set classes.
#   - Dev-monitor/dev-cal are derived from train_list_open_set.txt.
#   - Final acceptance should use official full-trial and strict profile evaluation.

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_cnceleb_finetune_from_vox2.sh /path/to/CN-Celeb_flac [vox2_pretrain_checkpoint] [target_epochs]"
  exit 1
fi

CN_ROOT="$1"
VOX2_CKPT="${2:-checkpoints/resnet_v7_vox2_aac12_pt_best.pth}"
TARGET_EPOCHS="${3:-20}"

if [[ ! -d "${CN_ROOT}" ]]; then
  echo "ERROR: CN root not found: ${CN_ROOT}"
  exit 1
fi

if [[ ! -f "${VOX2_CKPT}" ]]; then
  echo "ERROR: Vox2 pretrained checkpoint not found: ${VOX2_CKPT}"
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

echo "=== Stage 2A: Generate dev-monitor/dev-cal splits ==="
python make_dev_monitor.py \
  --train-list "lists/train_list_open_set.txt" \
  --base-path "${CN_ROOT}" \
  --monitor-speaker-ratio 0.05 \
  --cal-speaker-ratio 0.02 \
  --seed 42 \
  --output-dir "lists/dev_splits"

echo "=== Stage 2B: Finetune on CN-Celeb with active crop + dev-monitor ==="
python train.py \
  --base-path "${CN_ROOT}" \
  --list-path "lists/dev_splits/train_list_main.txt" \
  --init-model "${VOX2_CKPT}" \
  --optimizer adamw \
  --train-sampler balanced_speaker \
  --train-sample-repeat-cap 3 \
  --epochs "${TARGET_EPOCHS}" \
  --batch-size 32 \
  --lr 5e-4 \
  --min-lr 1e-6 \
  --warmup-epochs 3 \
  --warmup-start-factor 0.05 \
  --aam-scale 30 \
  --aam-margin 0.20 \
  --max-frames 300 \
  --train-crop-mode active \
  --num-workers 6 \
  --monitor-base-path "${CN_ROOT}" \
  --monitor-trials "lists/dev_splits/monitor/trials.lst" \
  --monitor-enroll-list "lists/dev_splits/monitor/enroll.lst" \
  --monitor-limit 2000 \
  --monitor-sample-mode balanced \
  --preprocess-for-inference \
  --show-grad-norm \
  --metrics-csv "artifacts/results/resnet_v7_vox2ft_s1_metrics.csv" \
  --ckpt-prefix "resnet_v7_vox2ft_s1"
