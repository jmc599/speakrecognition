#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_vox2_aac12_pretrain.sh /path/to/voxceleb2/voxceleb2 [target_epochs]
#
# Example:
#   bash run_vox2_aac12_pretrain.sh /data/datasets/vox/voxceleb2/voxceleb2 10
#
# Notes:
#   - VOX2_ROOT must be the directory that directly contains aac1/ and aac2/.
#   - This stage does not use an official Vox verification monitor.
#   - "best" is selected by validation loss, not EER.
#   - First round defaults to a 10-epoch probe.

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_vox2_aac12_pretrain.sh /path/to/voxceleb2/voxceleb2 [target_epochs]"
  exit 1
fi

VOX2_ROOT="$1"
TARGET_EPOCHS="${2:-10}"

if [[ ! -d "${VOX2_ROOT}" ]]; then
  echo "ERROR: Vox2 root not found: ${VOX2_ROOT}"
  exit 1
fi

if [[ ! -d "${VOX2_ROOT}/aac1" || ! -d "${VOX2_ROOT}/aac2" ]]; then
  echo "ERROR: Vox2 root must directly contain aac1/ and aac2/: ${VOX2_ROOT}"
  exit 1
fi

if [[ ! -f "lists/train_list_vox2_aac12.txt" ]]; then
  echo "ERROR: missing list lists/train_list_vox2_aac12.txt"
  exit 1
fi

python train.py \
  --base-path "${VOX2_ROOT}" \
  --list-path "lists/train_list_vox2_aac12.txt" \
  --optimizer adamw \
  --train-sampler balanced_speaker \
  --train-sample-repeat-cap 5 \
  --epochs "${TARGET_EPOCHS}" \
  --batch-size 32 \
  --lr 1e-3 \
  --min-lr 1e-6 \
  --warmup-epochs 3 \
  --warmup-start-factor 0.1 \
  --aam-scale 30 \
  --aam-margin 0.20 \
  --max-frames 300 \
  --train-crop-mode random \
  --num-workers 6 \
  --preprocess-for-inference \
  --show-grad-norm \
  --metrics-csv "artifacts/results/resnet_v7_vox2_aac12_pt_e${TARGET_EPOCHS}_metrics.csv" \
  --ckpt-prefix "resnet_v7_vox2_aac12_pt"
