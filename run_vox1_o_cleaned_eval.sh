#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_vox1_o_cleaned_eval.sh /path/to/voxceleb1 /path/to/vox1_o_cleaned.txt /path/to/checkpoint [limit]
#
# Notes:
#   - Supports official-style three-column trial files: "0/1 audio_a audio_b"
#   - Auto-selects base-path:
#       id... paths   -> <VOX1_ROOT>/wav
#       wav/id...     -> <VOX1_ROOT>
#   - This benchmark is for Stage 1 reporting only, not the final project result.

if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_vox1_o_cleaned_eval.sh /path/to/voxceleb1 /path/to/vox1_o_cleaned.txt /path/to/checkpoint [limit]"
  exit 1
fi

VOX1_ROOT="$1"
TRIALS_PATH="$2"
CKPT_PATH="$3"
LIMIT="${4:-0}"

[[ -d "${VOX1_ROOT}" ]] || { echo "ERROR: Vox1 root not found: ${VOX1_ROOT}"; exit 1; }
[[ -f "${TRIALS_PATH}" ]] || { echo "ERROR: trial list not found: ${TRIALS_PATH}"; exit 1; }
[[ -f "${CKPT_PATH}" ]] || { echo "ERROR: checkpoint not found: ${CKPT_PATH}"; exit 1; }

FIRST_LINE="$(grep -m 1 -v '^[[:space:]]*$' "${TRIALS_PATH}" || true)"
[[ -n "${FIRST_LINE}" ]] || { echo "ERROR: empty trial list: ${TRIALS_PATH}"; exit 1; }

FIRST_PATH="$(awk '{print $2}' <<< "${FIRST_LINE}")"
if [[ "${FIRST_PATH}" == wav/* ]]; then
  BASE_PATH="${VOX1_ROOT}"
elif [[ "${FIRST_PATH}" == id* ]]; then
  BASE_PATH="${VOX1_ROOT}/wav"
else
  echo "ERROR: unsupported trial path prefix in first line: ${FIRST_LINE}"
  exit 1
fi

CKPT_NAME="$(basename "${CKPT_PATH}")"
CKPT_STEM="${CKPT_NAME%.pth}"
OUTPUT_PATH="artifacts/results/${CKPT_STEM}_vox1_o_cleaned_scores.csv"

echo "Trial file: ${TRIALS_PATH}"
echo "Detected base-path: ${BASE_PATH}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Limit: ${LIMIT}"
echo "Output: ${OUTPUT_PATH}"

python score_trial_list.py \
  --checkpoint "${CKPT_PATH}" \
  --base-path "${BASE_PATH}" \
  --trials "${TRIALS_PATH}" \
  --output "${OUTPUT_PATH}" \
  --max-frames 300 \
  --preprocess-for-inference \
  --limit "${LIMIT}"
