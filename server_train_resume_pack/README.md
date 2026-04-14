# server_train_resume_pack

Minimal CN-Celeb training pack for Linux server.

## 1) Environment

Recommended: Python 3.11 + GPU PyTorch cu118.

Install minimal runtime deps (if not already installed):

```bash
python -m pip install --no-cache-dir torch==2.4.1+cu118 torchaudio==2.4.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy tqdm scipy
```

Optional:

```bash
python -m pip install tensorboard
python -m pip install av torchcodec
```

Quick verify:

```bash
python verify_env.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 2) Dataset Layout

`CN_ROOT` must contain:

- `data/`
- `eval/lists/trials.lst`
- `eval/lists/enroll.lst`

Example:

`~/cjm/cn_celeb/CN-Celeb_flac`

## 3) Train Entrypoints

### A. Resume your existing checkpoint (epoch10 -> epoch20)

```bash
bash resume_cnceleb_from_epoch10.sh ~/cjm/cn_celeb/CN-Celeb_flac 20
```

### B. Finetune from WeSpeaker pretrained weights

Default model path: `checkpoints/wespeaker/model_5.pt`

```bash
bash finetune_cnceleb_from_wespeaker.sh ~/cjm/cn_celeb/CN-Celeb_flac 20 checkpoints/wespeaker/model_5.pt
```

If model is missing:

```bash
mkdir -p checkpoints/wespeaker
wget -O checkpoints/wespeaker/model_5.pt \
  https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM/resolve/main/model_5.pt
```

## 4) Outputs

- Checkpoints: `checkpoints/*`
- Metrics CSV:
  - resume: `artifacts/results/resnet_v7_open_s1_metrics.csv`
  - wespeaker-init: `artifacts/results/resnet_v7_wes_init_s1_metrics.csv`
