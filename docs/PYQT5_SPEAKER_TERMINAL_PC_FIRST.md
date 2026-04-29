# PyQt5 Speaker Terminal PC-First Guide

This GUI flow now uses `resnet_v7_vox2ft_s1_latest` as the working baseline.

## Current baseline

- PyTorch checkpoint: `checkpoints/resnet_v7_vox2ft_s1_latest.pth`
- PC reference ONNX: `checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx`
- Remote RKNN target name: `/root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn`
- Threshold policy: `eer`

If these files are not present under `checkpoints/`, restore them from the returned bundle before running the GUI.

## 1. Install GUI dependencies on Windows

```bash
pip install -r requirements_gui.txt
```

The GUI also expects the normal local inference dependencies:

- `numpy`
- `torch`
- `torchaudio`
- an audio decoder path that can read `wav` and optional `m4a/mp3`

## 2. Export the fixed ONNX reference

Use the baseline checkpoint and keep the reference shape at `max_frames=300`:

```bash
python export_onnx.py \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --fixed-frames \
  --max-frames 300 \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx
```

## 3. Calibrate a threshold before GUI verification

The GUI no longer inherits old thresholds. Verification is blocked until the active backend has an explicit threshold.

Local PyTorch calibration example:

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

Local ONNX calibration example:

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

## 4. Launch the GUI

```bash
python speaker_identity_terminal.py
```

The default GUI config now points to the `vox2ft_s1_latest` artifact names. If the active backend has no threshold yet, the Verify tab will refuse to run until you calibrate or enter one manually.

## 5. Smoke scope for this phase

Use `Local ONNX` for the PC smoke test.

- Run at least 5 same-speaker trials.
- Run at least 5 obvious different-speaker trials.
- Do not re-fit thresholds during smoke.
- Do not use `Remote RKNN` for thresholded verification until RKNN parity has been checked and `remote_rknn_threshold` has been set explicitly.

## 6. Board-side preparation

The board currently has no microphone, so the near-real deployment path is:

`PC recording -> send WAV to board -> board preprocess + RKNN -> return result to PC GUI`

Before that phase, copy these files to `/root/models/` on the board:

- `resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn`
- `rknn_embed_once.py`

Do not assign a remote threshold from the old `0.187723` value.
