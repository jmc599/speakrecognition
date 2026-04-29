# RK3588 RKNN v1.6.0 Workflow

Use this workflow when the board runtime remains at `librknnrt 1.5.2` and the
current official toolkit wheel you can access is `rknn_toolkit2 1.6.0`.

## 1. Ubuntu conversion environment

Create and activate a dedicated Python 3.10 environment:

```bash
python3.10 -m venv ~/rknn160_venv
source ~/rknn160_venv/bin/activate
```

Install the pinned base dependencies from this repo:

```bash
pip install --upgrade pip
pip install -r /media/sf_speakerreg/requirements_rknn160.txt
```

This file tracks the official `1.6.0` requirements for the ONNX conversion
path. It intentionally omits `torch==1.13.1` and `tensorflow==2.8.0` because
we convert from an already-exported ONNX model. Only install those two if the
toolkit import explicitly reports a missing frontend dependency.
It also pins `numpy<2` because `onnxruntime==1.16.0` in this flow is not
compatible with NumPy 2.x.

Download the official `rknn_toolkit2 1.6.0` `cp310 x86_64` wheel on Windows,
copy it into `F:\speakerreg`, then install it from Ubuntu:

```bash
pip install "/media/sf_speakerreg/rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl"
python -c "from rknn.api import RKNN; print('rknn toolkit 1.6.0 ok')"
```

## 2. Rebuild the RKNN model

Try the existing fixed-shape ONNX first:

```bash
cd /media/sf_speakerreg
python convert_to_rknn.py
```

This writes:

```text
checkpoints/resnet_v7_best_fixed_rt160.rknn
```

If `load_onnx` fails in toolkit `1.6.0`, rebuild a lower-opset ONNX and retry:

```bash
python export_onnx.py \
  --checkpoint checkpoints/resnet_v7_best.pth \
  --output checkpoints/resnet_v7_best_fixed_op12.onnx \
  --normalize-output \
  --fixed-frames \
  --opset 12

python convert_to_rknn.py \
  --onnx-model checkpoints/resnet_v7_best_fixed_op12.onnx \
  --output checkpoints/resnet_v7_best_fixed_rt160.rknn
```

If runtime `1.5.2` reports unsupported `ReduceL2`, export a non-normalized ONNX
and rebuild the final deployment model:

```bash
python export_onnx.py \
  --checkpoint checkpoints/resnet_v7_best.pth \
  --output checkpoints/resnet_v7_best_fixed_nonorm_op12.onnx \
  --fixed-frames \
  --opset 12

python convert_to_rknn.py \
  --onnx-model checkpoints/resnet_v7_best_fixed_nonorm_op12.onnx \
  --output checkpoints/resnet_v7_best_fixed_rt160_nonorm.rknn
```

This `*_rt160_nonorm.rknn` artifact is the final validated deployment model for
the current board/runtime combination.

## 3. Consistency check before any threshold work

Generate the reference embeddings on Ubuntu:

```bash
python export_embedding_pytorch.py test_feature.npy --output embedding_pytorch.npy
python export_embedding_onnx.py test_feature.npy \
  --model checkpoints/resnet_v7_best_fixed.onnx \
  --output embedding_onnx.npy
```

Copy the new `.rknn` and `rknn_export_embedding.py` to the board. On the board:

```bash
python3 rknn_export_embedding.py test_feature.npy \
  --model resnet_v7_best_fixed_rt160_nonorm.rknn \
  --output embedding_rknn.npy
```

Copy `embedding_rknn.npy` back to Ubuntu, then compare:

```bash
python compare_embeddings.py
```

Required pass condition:

```text
pt vs rknn: cos >= 0.999 and l2 <= 1e-3
```

Do not run threshold tuning or batch evaluation until this passes.

## 4. Batch verification only after consistency passes

On the board:

```bash
python3 rknn_batch_verify.py \
  --manifest /root/models/rknn_eval_set_100/pairs.csv \
  --model /root/models/resnet_v7_best_fixed_rt160_nonorm.rknn \
  --base-dir /root/models/rknn_eval_set_100 \
  --save-results /root/models/rknn_eval_set_100/results.csv
```

Back on Ubuntu, sweep the best threshold:

```bash
python sweep_threshold.py --results /media/sf_speakerreg/results.csv
```

Current validated threshold on the 1000-pair evaluation set:

```text
0.187723
```
