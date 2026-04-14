#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_server_env.sh [env_name]
#
# Example:
#   bash setup_server_env.sh speaker311

ENV_NAME="${1:-speaker311}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIFORGE_CONDA="${HOME}/miniforge3/bin/conda"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "${MINIFORGE_CONDA}" ]]; then
    export PATH="${HOME}/miniforge3/bin:${PATH}"
    echo "conda not in PATH, using ${MINIFORGE_CONDA}"
  else
    echo "ERROR: conda not found."
    echo "Install Miniforge first (no sudo required):"
    echo "  cd ~"
    echo "  wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "  bash Miniforge3.sh -b -p ~/miniforge3"
    echo "  source ~/miniforge3/bin/activate"
    exit 1
  fi
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists, reuse it."
else
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
  echo "NVIDIA GPU detected, install CUDA PyTorch from: ${TORCH_INDEX_URL}"
  python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
else
  echo "No NVIDIA GPU detected, install CPU PyTorch..."
  python -m pip install torch torchvision torchaudio
fi

if [[ ! -f "${PROJECT_ROOT}/requirements_train_min.txt" ]]; then
  echo "ERROR: requirements_train_min.txt not found in ${PROJECT_ROOT}"
  exit 1
fi
python -m pip install -r "${PROJECT_ROOT}/requirements_train_min.txt"

if [[ "${INSTALL_OPTIONAL_CODECS:-0}" == "1" ]]; then
  echo "Installing optional audio codecs (av, torchcodec)..."
  python -m pip install av torchcodec || true
fi

python - <<'PY'
import torch, torchaudio, matplotlib
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("matplotlib:", matplotlib.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Environment setup done. Activate with: conda activate ${ENV_NAME}"
echo "Optional codecs (if needed): INSTALL_OPTIONAL_CODECS=1 bash setup_server_env.sh ${ENV_NAME}"
