#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_vox2_cn_clean_mainline.sh /path/to/voxceleb2 /path/to/CN-Celeb_flac [strict_feature_root]
#   bash run_vox2_cn_clean_mainline.sh /path/to/voxceleb2 /path/to/CN-Celeb_flac \
#     --continue-after-stage1 \
#     --stage1-epochs 30 \
#     --stage2-epochs 20 \
#     --strict-feature-root /path/to/features
#   bash run_vox2_cn_clean_mainline.sh /path/to/voxceleb2 /path/to/CN-Celeb_flac \
#     --skip-stage1 \
#     --continue-after-stage1
#
# Default behavior:
#   - Run Vox Stage 1 only, then stop for manual review.
#   - This avoids silently continuing into CN finetune before you inspect the
#     Stage 1 probe curve and decide whether to extend training.

usage() {
  cat <<'EOF'
Usage:
  bash run_vox2_cn_clean_mainline.sh /path/to/voxceleb2 /path/to/CN-Celeb_flac [strict_feature_root]
  bash run_vox2_cn_clean_mainline.sh /path/to/voxceleb2 /path/to/CN-Celeb_flac [options]

Options:
  --stage1-epochs N         Target epochs for Vox Stage 1. Default: 10
  --stage2-epochs N         Target epochs for CN Stage 2. Default: 20
  --strict-feature-root P   Optional strict-profile feature root
  --continue-after-stage1   Continue into CN finetune and final evaluation
  --skip-stage1             Reuse existing Stage 1 checkpoint and skip Vox training
  -h, --help                Show this help message
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

VOX2_ROOT="$1"
CN_ROOT="$2"
shift 2

STRICT_FEATURE_ROOT=""
STAGE1_EPOCHS=10
STAGE2_EPOCHS=20
CONTINUE_AFTER_STAGE1=0
SKIP_STAGE1=0

if [[ $# -gt 0 && "${1}" != --* ]]; then
  STRICT_FEATURE_ROOT="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1-epochs)
      [[ $# -ge 2 ]] || { echo "ERROR: --stage1-epochs requires a value"; exit 1; }
      STAGE1_EPOCHS="$2"
      shift 2
      ;;
    --stage2-epochs)
      [[ $# -ge 2 ]] || { echo "ERROR: --stage2-epochs requires a value"; exit 1; }
      STAGE2_EPOCHS="$2"
      shift 2
      ;;
    --strict-feature-root)
      [[ $# -ge 2 ]] || { echo "ERROR: --strict-feature-root requires a value"; exit 1; }
      STRICT_FEATURE_ROOT="$2"
      shift 2
      ;;
    --continue-after-stage1)
      CONTINUE_AFTER_STAGE1=1
      shift
      ;;
    --skip-stage1)
      SKIP_STAGE1=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

[[ -d "${VOX2_ROOT}" ]] || { echo "ERROR: Vox2 root not found: ${VOX2_ROOT}"; exit 1; }
[[ -d "${VOX2_ROOT}/aac1" && -d "${VOX2_ROOT}/aac2" ]] || {
  echo "ERROR: Vox2 root must directly contain aac1/ and aac2/: ${VOX2_ROOT}"
  exit 1
}
[[ -d "${CN_ROOT}" ]] || { echo "ERROR: CN root not found: ${CN_ROOT}"; exit 1; }
[[ -f "lists/train_list_vox2_aac12.txt" ]] || { echo "ERROR: missing lists/train_list_vox2_aac12.txt"; exit 1; }
[[ -f "lists/train_list_open_set.txt" ]] || { echo "ERROR: missing lists/train_list_open_set.txt"; exit 1; }
[[ -f "${CN_ROOT}/eval/lists/trials.lst" ]] || { echo "ERROR: missing ${CN_ROOT}/eval/lists/trials.lst"; exit 1; }
[[ -f "${CN_ROOT}/eval/lists/enroll.lst" ]] || { echo "ERROR: missing ${CN_ROOT}/eval/lists/enroll.lst"; exit 1; }

if [[ "${SKIP_STAGE1}" -eq 0 ]]; then
  echo "=== Stage 1: Vox aac12 probe (${STAGE1_EPOCHS} epochs) ==="
  bash run_vox2_aac12_pretrain.sh "${VOX2_ROOT}" "${STAGE1_EPOCHS}"
else
  echo "=== Stage 1 skipped: reusing existing Vox checkpoint ==="
fi

[[ -f "checkpoints/resnet_v7_vox2_aac12_pt_best.pth" ]] || {
  echo "ERROR: expected checkpoint checkpoints/resnet_v7_vox2_aac12_pt_best.pth after Stage 1"
  exit 1
}

if [[ "${CONTINUE_AFTER_STAGE1}" -ne 1 ]]; then
  echo "=== Stage 1 completed. Stopping before CN finetune by design. ==="
  echo "Review artifacts/results/resnet_v7_vox2_aac12_pt_e${STAGE1_EPOCHS}_metrics.csv"
  echo "If Stage 1 looks good and you want to continue, rerun with:"
  echo "  bash run_vox2_cn_clean_mainline.sh \"${VOX2_ROOT}\" \"${CN_ROOT}\" --skip-stage1 --continue-after-stage1 --stage2-epochs ${STAGE2_EPOCHS}"
  if [[ -n "${STRICT_FEATURE_ROOT}" ]]; then
    echo "  bash run_vox2_cn_clean_mainline.sh \"${VOX2_ROOT}\" \"${CN_ROOT}\" --skip-stage1 --continue-after-stage1 --stage2-epochs ${STAGE2_EPOCHS} --strict-feature-root \"${STRICT_FEATURE_ROOT}\""
  fi
  exit 0
fi

echo "=== Stage 2: CN-Celeb finetune with dev-monitor + active crop ==="
bash run_cnceleb_finetune_from_vox2.sh "${CN_ROOT}" "checkpoints/resnet_v7_vox2_aac12_pt_best.pth" "${STAGE2_EPOCHS}"

[[ -f "checkpoints/resnet_v7_vox2ft_s1_best.pth" ]] || {
  echo "ERROR: expected checkpoint checkpoints/resnet_v7_vox2ft_s1_best.pth after Stage 2"
  exit 1
}

echo "=== Stage 3A: Official full-trial (best checkpoint only) ==="
python evaluate_candidate_checkpoints.py \
  --checkpoint "checkpoints/resnet_v7_vox2ft_s1_best.pth" \
  --base-path "${CN_ROOT}/eval" \
  --trials "${CN_ROOT}/eval/lists/trials.lst" \
  --enroll-list "${CN_ROOT}/eval/lists/enroll.lst" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output-json "artifacts/results/resnet_v7_vox2ft_s1_eval.json"

echo "=== Stage 3B: Fit dev-cal thresholds ==="
python score_trial_list.py \
  --checkpoint "checkpoints/resnet_v7_vox2ft_s1_best.pth" \
  --base-path "${CN_ROOT}" \
  --trials "lists/dev_splits/cal/trials.lst" \
  --enroll-list "lists/dev_splits/cal/enroll.lst" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_vox2ft_s1_devcal_scores.csv"

python calibrate_profile_results.py \
  --results "artifacts/results/resnet_v7_vox2ft_s1_devcal_scores.csv" \
  --checkpoint "checkpoints/resnet_v7_vox2ft_s1_best.pth" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_vox2ft_s1_devcal_calibration.json"

if [[ -n "${STRICT_FEATURE_ROOT}" ]]; then
  [[ -d "${STRICT_FEATURE_ROOT}" ]] || {
    echo "ERROR: strict feature root not found: ${STRICT_FEATURE_ROOT}"
    exit 1
  }

  echo "=== Stage 3C: Strict profile raw ==="
  python pytorch_profile_verify.py \
    --feature-root "${STRICT_FEATURE_ROOT}" \
    --checkpoint "checkpoints/resnet_v7_vox2ft_s1_best.pth" \
    --save-results "artifacts/results/resnet_v7_vox2ft_s1_strict_raw.csv" \
    --max-frames 300 \
    --preprocess-for-inference

  echo "=== Stage 3D: Strict profile calibrated with dev-cal thresholds ==="
  python calibrate_profile_results.py \
    --results "artifacts/results/resnet_v7_vox2ft_s1_strict_raw.csv" \
    --reference-calibration "artifacts/results/resnet_v7_vox2ft_s1_devcal_calibration.json" \
    --checkpoint "checkpoints/resnet_v7_vox2ft_s1_best.pth" \
    --max-frames 300 \
    --preprocess-for-inference \
    --output "artifacts/results/resnet_v7_vox2ft_s1_strict_calibrated.json"
else
  echo "=== Stage 3C skipped: strict feature root not provided ==="
fi

echo "Done: Route 3 clean mainline completed."
