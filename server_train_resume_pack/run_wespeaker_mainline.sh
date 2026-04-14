#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_wespeaker_mainline.sh /path/to/CN-Celeb_flac [wespeaker_model_path] [strict_feature_root]
#
# Example:
#   bash run_wespeaker_mainline.sh \
#     /home/ubuntu/cjm/cn_celeb/cn-celeb_v2/CN-Celeb_flac \
#     checkpoints/wespeaker/model_5.pt \
#     rknn_eval_set_official_eval_inf/features

CN_ROOT="${1:?Usage: bash run_wespeaker_mainline.sh /path/to/CN-Celeb_flac [wespeaker_model_path] [strict_feature_root]}"
WESPK_MODEL="${2:-checkpoints/wespeaker/model_5.pt}"
STRICT_FEATURE_ROOT="${3:-rknn_eval_set_official_eval_inf/features}"

[[ -d "${CN_ROOT}" ]] || { echo "ERROR: CN root not found: ${CN_ROOT}"; exit 1; }
[[ -f "${WESPK_MODEL}" ]] || { echo "ERROR: WeSpeaker model not found: ${WESPK_MODEL}"; exit 1; }
[[ -f "lists/train_list_open_set.txt" ]] || { echo "ERROR: missing lists/train_list_open_set.txt"; exit 1; }
[[ -d "${STRICT_FEATURE_ROOT}" ]] || { echo "ERROR: strict feature root not found: ${STRICT_FEATURE_ROOT}"; exit 1; }
[[ -f "${CN_ROOT}/eval/lists/trials.lst" ]] || { echo "ERROR: missing ${CN_ROOT}/eval/lists/trials.lst"; exit 1; }
[[ -f "${CN_ROOT}/eval/lists/enroll.lst" ]] || { echo "ERROR: missing ${CN_ROOT}/eval/lists/enroll.lst"; exit 1; }

echo "=== Stage 1: Generate dev-monitor/dev-cal splits ==="
python make_dev_monitor.py \
  --train-list "lists/train_list_open_set.txt" \
  --base-path "${CN_ROOT}" \
  --monitor-speaker-ratio 0.05 \
  --cal-speaker-ratio 0.02 \
  --seed 42 \
  --output-dir "lists/dev_splits"

echo "=== Stage 2A: Train WeSpeaker mainline ==="
python train.py \
  --base-path "${CN_ROOT}" \
  --list-path "lists/dev_splits/train_list_main.txt" \
  --init-model "${WESPK_MODEL}" \
  --optimizer adamw \
  --train-sampler balanced_speaker \
  --train-sample-repeat-cap 3 \
  --epochs 20 \
  --batch-size 32 \
  --lr 5e-4 \
  --min-lr 1e-6 \
  --warmup-epochs 3 \
  --warmup-start-factor 0.05 \
  --aam-scale 30 \
  --aam-margin 0.20 \
  --max-frames 300 \
  --num-workers 6 \
  --monitor-base-path "${CN_ROOT}" \
  --monitor-trials "lists/dev_splits/monitor/trials.lst" \
  --monitor-enroll-list "lists/dev_splits/monitor/enroll.lst" \
  --monitor-limit 2000 \
  --monitor-sample-mode balanced \
  --preprocess-for-inference \
  --show-grad-norm \
  --metrics-csv "artifacts/results/resnet_v7_wes_mainline_metrics.csv" \
  --ckpt-prefix "resnet_v7_wes_mainline"

echo "=== Stage 2B: Train no-init control ==="
python train.py \
  --base-path "${CN_ROOT}" \
  --list-path "lists/dev_splits/train_list_main.txt" \
  --optimizer adamw \
  --train-sampler balanced_speaker \
  --train-sample-repeat-cap 3 \
  --epochs 20 \
  --batch-size 32 \
  --lr 5e-4 \
  --min-lr 1e-6 \
  --warmup-epochs 3 \
  --warmup-start-factor 0.05 \
  --aam-scale 30 \
  --aam-margin 0.20 \
  --max-frames 300 \
  --num-workers 6 \
  --monitor-base-path "${CN_ROOT}" \
  --monitor-trials "lists/dev_splits/monitor/trials.lst" \
  --monitor-enroll-list "lists/dev_splits/monitor/enroll.lst" \
  --monitor-limit 2000 \
  --monitor-sample-mode balanced \
  --preprocess-for-inference \
  --show-grad-norm \
  --metrics-csv "artifacts/results/resnet_v7_ctrl_mainline_metrics.csv" \
  --ckpt-prefix "resnet_v7_ctrl_mainline"

echo "=== Stage 3A: Official full-trial (best checkpoints only) ==="
python evaluate_candidate_checkpoints.py \
  --checkpoint "checkpoints/resnet_v7_wes_mainline_best.pth" \
  --checkpoint "checkpoints/resnet_v7_ctrl_mainline_best.pth" \
  --base-path "${CN_ROOT}/eval" \
  --trials "${CN_ROOT}/eval/lists/trials.lst" \
  --enroll-list "${CN_ROOT}/eval/lists/enroll.lst" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output-json "artifacts/results/resnet_v7_wes_vs_ctrl_full_eval.json"

echo "=== Stage 3B: Fit dev-cal thresholds (WeSpeaker mainline) ==="
python score_trial_list.py \
  --checkpoint "checkpoints/resnet_v7_wes_mainline_best.pth" \
  --base-path "${CN_ROOT}" \
  --trials "lists/dev_splits/cal/trials.lst" \
  --enroll-list "lists/dev_splits/cal/enroll.lst" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_wes_mainline_devcal_scores.csv"

python calibrate_profile_results.py \
  --results "artifacts/results/resnet_v7_wes_mainline_devcal_scores.csv" \
  --checkpoint "checkpoints/resnet_v7_wes_mainline_best.pth" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_wes_mainline_devcal_calibration.json"

echo "=== Stage 3C: Fit dev-cal thresholds (no-init control) ==="
python score_trial_list.py \
  --checkpoint "checkpoints/resnet_v7_ctrl_mainline_best.pth" \
  --base-path "${CN_ROOT}" \
  --trials "lists/dev_splits/cal/trials.lst" \
  --enroll-list "lists/dev_splits/cal/enroll.lst" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_ctrl_mainline_devcal_scores.csv"

python calibrate_profile_results.py \
  --results "artifacts/results/resnet_v7_ctrl_mainline_devcal_scores.csv" \
  --checkpoint "checkpoints/resnet_v7_ctrl_mainline_best.pth" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_ctrl_mainline_devcal_calibration.json"

echo "=== Stage 3D: Strict profile raw (WeSpeaker mainline) ==="
python pytorch_profile_verify.py \
  --feature-root "${STRICT_FEATURE_ROOT}" \
  --checkpoint "checkpoints/resnet_v7_wes_mainline_best.pth" \
  --save-results "artifacts/results/resnet_v7_wes_mainline_strict_raw.csv" \
  --max-frames 300 \
  --preprocess-for-inference

python calibrate_profile_results.py \
  --results "artifacts/results/resnet_v7_wes_mainline_strict_raw.csv" \
  --reference-calibration "artifacts/results/resnet_v7_wes_mainline_devcal_calibration.json" \
  --checkpoint "checkpoints/resnet_v7_wes_mainline_best.pth" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_wes_mainline_strict_calibrated.json"

echo "=== Stage 3E: Strict profile raw (no-init control) ==="
python pytorch_profile_verify.py \
  --feature-root "${STRICT_FEATURE_ROOT}" \
  --checkpoint "checkpoints/resnet_v7_ctrl_mainline_best.pth" \
  --save-results "artifacts/results/resnet_v7_ctrl_mainline_strict_raw.csv" \
  --max-frames 300 \
  --preprocess-for-inference

python calibrate_profile_results.py \
  --results "artifacts/results/resnet_v7_ctrl_mainline_strict_raw.csv" \
  --reference-calibration "artifacts/results/resnet_v7_ctrl_mainline_devcal_calibration.json" \
  --checkpoint "checkpoints/resnet_v7_ctrl_mainline_best.pth" \
  --max-frames 300 \
  --preprocess-for-inference \
  --output "artifacts/results/resnet_v7_ctrl_mainline_strict_calibrated.json"

echo "Done: WeSpeaker mainline and paired control completed."
