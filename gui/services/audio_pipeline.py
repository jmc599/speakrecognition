from pathlib import Path
from uuid import uuid4

import soundfile as sf

from gui.services.paths import local_tmp_dir


def normalize_audio_file(source_path, prefix="audio"):
    try:
        from utils.audio_io import load_audio_mono_16k
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "音频预处理依赖缺失。请在当前环境安装 torch/torchaudio/PyAV。"
        ) from exc

    waveform, sample_rate = load_audio_mono_16k(source_path)
    output_path = Path(local_tmp_dir()) / f"{prefix}_{uuid4().hex}_16k.wav"
    sf.write(str(output_path), waveform.squeeze(0).numpy(), sample_rate)
    return str(output_path)
