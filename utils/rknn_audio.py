from pathlib import Path
import wave

import numpy as np

from utils.inference_audio import preprocess_waveform_for_inference
from utils.legacy_feature import DEFAULT_MAX_FRAMES, LEGACY_NUM_MELS


SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = LEGACY_NUM_MELS
MAX_FRAMES = DEFAULT_MAX_FRAMES
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2.0


def _read_wav(audio_path):
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    with wave.open(str(audio_path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif sample_width == 4:
        try:
            data = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
        except ValueError as exc:
            raise RuntimeError(f"Unsupported WAV format: {audio_path}") from exc
    else:
        raise RuntimeError(
            f"Unsupported WAV sample width {sample_width} bytes: {audio_path}"
        )

    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)

    return data.astype(np.float32, copy=False), int(sample_rate)


def load_waveform(audio_path, preprocess_for_inference=True):
    waveform, sr = _read_wav(audio_path)
    if sr != SAMPLE_RATE:
        raise RuntimeError(
            f"Expected 16k WAV input for board-side RKNN runner, got sample_rate={sr}: {audio_path}"
        )
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(f"Failed to decode audio: {audio_path}")
    if preprocess_for_inference:
        waveform, _ = preprocess_waveform_for_inference(
            waveform,
            sample_rate=sr,
            trim_silence=True,
            normalize_volume=True,
        )
    return waveform.astype(np.float32, copy=False), sr


def _hz_to_mel_htk(freq_hz):
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def _mel_to_hz_htk(mels):
    return 700.0 * (10.0 ** (np.asarray(mels, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank():
    n_freqs = N_FFT // 2 + 1
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, n_freqs, dtype=np.float64)

    mel_min = _hz_to_mel_htk(F_MIN)
    mel_max = _hz_to_mel_htk(F_MAX)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2, dtype=np.float64)
    hz_points = _mel_to_hz_htk(mel_points)

    fb = np.zeros((N_MELS, n_freqs), dtype=np.float32)
    for i in range(N_MELS):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        if center <= left or right <= center:
            continue

        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        fb[i, left_mask] = ((fft_freqs[left_mask] - left) / (center - left)).astype(np.float32)
        fb[i, right_mask] = ((right - fft_freqs[right_mask]) / (right - center)).astype(np.float32)

    return fb


_MEL_FILTERBANK = _mel_filterbank()
_HANN_WINDOW = np.hanning(WIN_LENGTH).astype(np.float32)


def _stft_power(waveform):
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    pad = N_FFT // 2
    x = np.pad(x, (pad, pad), mode='reflect')

    if x.size < WIN_LENGTH:
        x = np.pad(x, (0, WIN_LENGTH - x.size), mode='constant')

    frame_count = 1 + (x.size - WIN_LENGTH) // HOP_LENGTH
    frames = np.empty((frame_count, WIN_LENGTH), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * HOP_LENGTH
        frames[idx] = x[start : start + WIN_LENGTH]

    windowed = frames * _HANN_WINDOW[None, :]
    spectrum = np.fft.rfft(windowed, n=N_FFT, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    return power.T


def waveform_to_logmel(waveform):
    power = _stft_power(waveform)
    mel = np.matmul(_MEL_FILTERBANK, power)
    return np.log(mel + 1e-6).astype(np.float32)


def make_chunks(logmel, max_frames=MAX_FRAMES, num_eval=5):
    total_frames = logmel.shape[1]
    if total_frames <= max_frames:
        padded = np.pad(
            logmel,
            ((0, 0), (0, max_frames - total_frames)),
            mode='constant',
        )
        return padded[None, None, :, :]

    if num_eval <= 1:
        starts = [0]
    else:
        max_start = total_frames - max_frames
        starts = np.linspace(0, max_start, num=num_eval, dtype=np.int64).tolist()

    chunks = [logmel[:, start : start + max_frames] for start in starts]
    stacked = np.stack(chunks, axis=0).astype(np.float32)
    return stacked[:, None, :, :]


def l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def run_model(rknn_lite, batch_input):
    outputs = rknn_lite.inference(inputs=[batch_input])
    if not outputs:
        raise RuntimeError('RKNN inference returned no outputs.')
    return np.asarray(outputs[0], dtype=np.float32)


def merge_embedding_output(output):
    emb = np.asarray(output, dtype=np.float32)
    emb = l2_normalize(emb, axis=1)
    emb = emb.mean(axis=0, keepdims=True)
    return l2_normalize(emb, axis=1)


def extract_embedding(rknn_lite, audio_path, num_eval=5, preprocess_for_inference=True):
    waveform, _ = load_waveform(
        audio_path,
        preprocess_for_inference=preprocess_for_inference,
    )
    logmel = waveform_to_logmel(waveform)
    batch_input = make_chunks(logmel, max_frames=MAX_FRAMES, num_eval=num_eval)
    output = run_model(rknn_lite, batch_input)
    return merge_embedding_output(output)


def cosine_score(emb_a, emb_b):
    emb_a = np.asarray(emb_a, dtype=np.float32).reshape(1, -1)
    emb_b = np.asarray(emb_b, dtype=np.float32).reshape(1, -1)
    return float(np.sum(emb_a * emb_b, axis=1)[0])


def create_runtime(model_path, *, verbose=False, target=None):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f'RKNN model not found: {model_path}')

    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('rknnlite is required for RKNN audio inference.') from exc

    rknn_lite = RKNNLite(verbose=bool(verbose))
    ret = rknn_lite.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f'load_rknn failed: {ret}')
    init_kwargs = {}
    if target:
        init_kwargs['target'] = str(target)
    ret = rknn_lite.init_runtime(**init_kwargs)
    if ret != 0:
        raise RuntimeError(f'init_runtime failed: {ret}')
    return rknn_lite
