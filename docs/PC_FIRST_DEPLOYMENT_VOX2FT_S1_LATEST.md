# `resnet_v7_vox2ft_s1` PC-First Deployment

This document is the current implementation baseline.

## Scope

- Deployment baseline checkpoint: `checkpoints/resnet_v7_vox2ft_s1_latest.pth`
- Threshold policy: `eer`
- Fixed ONNX reference shape: `max_frames=300`
- PC-side verification comes first
- RKNN is phase 2 and remains parity-gated

## Phase 1: Record the baseline manifest

```bash
python scripts/write_pcfirst_manifest.py \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --role measured_reference
```

This writes a manifest that records:

- checkpoint filename
- existence and SHA-256
- expected fixed ONNX filename
- expected RT160 fallback RKNN filename
- current role and threshold policy

## Phase 2: Local PyTorch calibration

```bash
python pytorch_profile_verify.py \
  --feature-root <FEATURE_ROOT> \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --enroll-count 3 \
  --preprocess-for-inference \
  --threshold-policy eer \
  --save-results artifacts/results/results_pytorch_profile_latest.csv \
  --save-calibration artifacts/results/results_pytorch_profile_latest_calibration.json \
  --write-gui-config
```

Expected outcome:

- a results CSV
- a calibration JSON containing `results`, `checkpoint`, `max_frames`, `preprocess_for_inference`, `threshold_policy`, and `selected_threshold`
- GUI config updated for `local_pytorch_threshold`

## Phase 3: Export and validate the PC ONNX

Export:

```bash
python export_onnx.py \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --fixed-frames \
  --max-frames 300 \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx
```

Validate:

```bash
python onnx_profile_verify.py \
  --feature-root <FEATURE_ROOT> \
  --model checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx \
  --enroll-count 3 \
  --preprocess-for-inference \
  --threshold-policy eer \
  --save-results artifacts/results/results_onnx_profile_latest.csv \
  --save-calibration artifacts/results/results_onnx_profile_latest_calibration.json \
  --write-gui-config
```

Acceptance target:

- ONNX and PyTorch `EER` difference within `0.2` percentage points
- threshold difference preferably within `0.02` cosine

## Phase 4: GUI smoke on Windows

Launch:

```bash
python speaker_identity_terminal.py
```

Use the `Local ONNX` backend for this phase.

- run at least 5 same-speaker trials
- run at least 5 obvious different-speaker trials
- do not re-fit thresholds during smoke
- fix config or backend errors before moving on

## Phase 5: RKNN parity on Ubuntu and board

Use the RT160 workflow in [RKNN_RT160_PC_FIRST.md](RKNN_RT160_PC_FIRST.md).

Rules:

- `remote_rknn_threshold` remains unset until RKNN parity is checked
- reuse the ONNX threshold only if RKNN transfer looks stable
- otherwise fit a separate RKNN threshold and write it explicitly

## Phase 6: Near-real chain without board microphone

Deployment chain for now:

`PC record WAV -> board preprocess + RKNN + scoring -> PC display`

Validate:

- sample rate
- PCM to float scaling
- trim and padding behavior
- score consistency against the PC reference path

## What changed in code

- CLI defaults now point to `vox2ft_s1_latest` artifact names
- `export_onnx.py` and `convert_to_rknn.py` follow the selected checkpoint stem
- calibration JSON now stores explicit threshold policy and selected threshold
- GUI config no longer inherits legacy thresholds like `0.213456` or `0.187723`
- verification is blocked when the active backend has no threshold
