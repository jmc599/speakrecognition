# RK3588 RT160 PC-First Workflow

Use this workflow only after the PC-side `PyTorch -> ONNX -> GUI` path has passed for `resnet_v7_vox2ft_s1_latest`.

## Status

- Current measured reference checkpoint: `checkpoints/resnet_v7_vox2ft_s1_latest.pth`
- PC reference ONNX: `checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx`
- RKNN is still a parity-risk branch under `rknn_toolkit2 1.6.0` and board runtime `1.5.2`
- `remote_rknn_threshold` must stay unset until RKNN consistency is proven

## 1. Ubuntu conversion environment

```bash
python3.10 -m venv ~/rknn160_venv
source ~/rknn160_venv/bin/activate
pip install --upgrade pip
pip install -r /media/sf_speakerreg/requirements_rknn160.txt
pip install "/media/sf_speakerreg/rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl"
python -c "from rknn.api import RKNN; print('rknn toolkit 1.6.0 ok')"
```

## 2. Export the reference ONNX

Keep the fixed-shape reference export at `max_frames=300`:

```bash
cd /media/sf_speakerreg
python export_onnx.py \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --fixed-frames \
  --max-frames 300 \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx
```

## 3. Convert to RKNN on Ubuntu

Default RT160 conversion:

```bash
python convert_to_rknn.py \
  --onnx-model checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed_rt160.rknn
```

If toolkit `1.6.0` rejects the ONNX graph, rebuild an opset-12 export:

```bash
python export_onnx.py \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --fixed-frames \
  --max-frames 300 \
  --opset 12 \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed_op12.onnx

python convert_to_rknn.py \
  --onnx-model checkpoints/resnet_v7_vox2ft_s1_latest_fixed_op12.onnx \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed_rt160.rknn
```

If runtime `1.5.2` has trouble with `ReduceL2` or normalized output, switch to the fallback branch:

```bash
python export_onnx.py \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --fixed-frames \
  --max-frames 300 \
  --opset 12 \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed_nonorm_op12.onnx

python convert_to_rknn.py \
  --onnx-model checkpoints/resnet_v7_vox2ft_s1_latest_fixed_nonorm_op12.onnx \
  --output checkpoints/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn
```

That `*_rt160_nonorm.rknn` artifact is the expected deployment artifact for the current RT160 fallback path.

## 4. Embedding parity before any threshold work

Generate reference embeddings on Ubuntu:

```bash
python export_embedding_pytorch.py test_feature.npy \
  --checkpoint checkpoints/resnet_v7_vox2ft_s1_latest.pth \
  --output embedding_pytorch.npy

python export_embedding_onnx.py test_feature.npy \
  --model checkpoints/resnet_v7_vox2ft_s1_latest_fixed.onnx \
  --output embedding_onnx.npy
```

Copy the RKNN model and `rknn_export_embedding.py` to the board, then on the board:

```bash
python3 rknn_export_embedding.py test_feature.npy \
  --model /root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn \
  --output embedding_rknn.npy
```

Back on Ubuntu:

```bash
python compare_embeddings.py
```

Required pass condition:

```text
pt vs rknn: cos >= 0.999 and l2 <= 1e-3
```

Do not fit or transfer any RKNN threshold until this passes.

## 5. Batch/profile verification on RKNN

Run RKNN verification without a threshold first so the results stay policy-neutral:

```bash
python3 rknn_batch_verify.py \
  --manifest /root/models/rknn_eval_set/pairs.csv \
  --model /root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn \
  --base-dir /root/models/rknn_eval_set \
  --save-results /root/models/rknn_eval_set/results_rknn.csv
```

Pull the results back to Ubuntu and evaluate threshold transfer from the ONNX calibration:

```bash
python calibrate_profile_results.py \
  --results /media/sf_speakerreg/artifacts/results/results_rknn.csv \
  --reference-calibration /media/sf_speakerreg/artifacts/results/results_onnx_profile_latest_calibration.json \
  --checkpoint /root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn \
  --max-frames 300 \
  --preprocess-for-inference \
  --threshold-policy eer
```

If RKNN remains close enough to ONNX, you may reuse the ONNX pilot threshold. If it drifts materially, fit a dedicated remote threshold instead:

```bash
python calibrate_profile_results.py \
  --results /media/sf_speakerreg/artifacts/results/results_rknn.csv \
  --checkpoint /root/models/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn \
  --max-frames 300 \
  --preprocess-for-inference \
  --threshold-policy eer \
  --backend-mode remote_rknn \
  --write-gui-config
```

## 6. Important rules

- Do not inherit the old `0.187723` threshold.
- Do not assume RKNN can reuse the ONNX threshold until parity is checked.
- Do not do the main RKNN conversion on Windows. Use Ubuntu for conversion and the board for runtime validation.
