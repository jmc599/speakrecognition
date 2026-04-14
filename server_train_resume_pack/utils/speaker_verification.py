from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from core.model import ResNet34_SE
from utils.audio_io import load_audio_mono_16k
from utils.checkpoint_io import load_torch_checkpoint
from utils.legacy_feature import DEFAULT_MAX_FRAMES, LEGACY_NUM_MELS


_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


def resolve_audio_path(audio_path, base_path=""):
    candidates = []
    path = Path(audio_path)
    candidates.append(path)

    if base_path:
        candidates.append(Path(base_path) / audio_path)

    checked = []
    for candidate in candidates:
        checked.append(candidate)
        if candidate.is_file():
            return str(candidate)

        if candidate.suffix:
            stem = candidate.with_suffix("")
            for suffix in _AUDIO_SUFFIXES:
                alt = stem.with_suffix(suffix)
                checked.append(alt)
                if alt.is_file():
                    return str(alt)
        else:
            for suffix in _AUDIO_SUFFIXES:
                alt = candidate.with_suffix(suffix)
                checked.append(alt)
                if alt.is_file():
                    return str(alt)

    checked_str = ", ".join(str(p) for p in checked[:8])
    raise FileNotFoundError(f"Audio file not found: {audio_path}. Checked: {checked_str}")


def build_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=LEGACY_NUM_MELS,
    )


def load_audio_chunks(
    audio_path,
    mel_transform,
    max_frames=DEFAULT_MAX_FRAMES,
    num_eval=5,
    preprocess_for_inference=False,
):
    waveform, _ = load_audio_mono_16k(
        audio_path,
        for_inference=preprocess_for_inference,
    )

    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-6)

    _, _, total_frames = mel_spec.shape
    if total_frames <= max_frames:
        pad_len = max_frames - total_frames
        if pad_len > 0:
            mel_spec = F.pad(mel_spec, (0, pad_len))
        return mel_spec.unsqueeze(0)

    if num_eval <= 1:
        starts = [0]
    else:
        max_start = total_frames - max_frames
        starts = torch.linspace(0, max_start, steps=num_eval).long().tolist()

    chunks = [mel_spec[:, :, start : start + max_frames] for start in starts]
    return torch.stack(chunks, dim=0)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_embedding_dim(checkpoint):
    state_dict = _extract_state_dict(checkpoint)
    fc_weight = state_dict.get("fc.weight")
    if fc_weight is None:
        raise KeyError("Cannot infer embedding_dim from checkpoint: missing fc.weight")
    return int(fc_weight.shape[0])


def load_model(checkpoint_path, device, embedding_dim=None):
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    if embedding_dim is None:
        embedding_dim = infer_embedding_dim(checkpoint)

    model = ResNet34_SE(embedding_dim=embedding_dim, input_channels=1).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. "
            "The current ResNet model was changed and old checkpoints are not compatible. "
            "Train a new checkpoint with the updated model or switch back to the old model definition."
        ) from exc
    model.eval()
    return model, checkpoint


@torch.no_grad()
def extract_embedding(
    model,
    audio_path,
    device,
    mel_transform,
    base_path="",
    max_frames=DEFAULT_MAX_FRAMES,
    num_eval=5,
    preprocess_for_inference=False,
):
    resolved = resolve_audio_path(audio_path, base_path=base_path)
    chunks = load_audio_chunks(
        resolved,
        mel_transform=mel_transform,
        max_frames=max_frames,
        num_eval=num_eval,
        preprocess_for_inference=preprocess_for_inference,
    ).to(device)

    emb = model(chunks)
    emb = F.normalize(emb, dim=1)
    emb = emb.mean(dim=0, keepdim=True)
    emb = F.normalize(emb, dim=1)
    return emb.cpu(), resolved


@torch.no_grad()
def cosine_score(
    model,
    audio_a,
    audio_b,
    device,
    mel_transform,
    base_path="",
    max_frames=DEFAULT_MAX_FRAMES,
    num_eval=5,
    preprocess_for_inference=False,
):
    emb_a, path_a = extract_embedding(
        model,
        audio_a,
        device,
        mel_transform,
        base_path=base_path,
        max_frames=max_frames,
        num_eval=num_eval,
        preprocess_for_inference=preprocess_for_inference,
    )
    emb_b, path_b = extract_embedding(
        model,
        audio_b,
        device,
        mel_transform,
        base_path=base_path,
        max_frames=max_frames,
        num_eval=num_eval,
        preprocess_for_inference=preprocess_for_inference,
    )
    score = F.cosine_similarity(emb_a, emb_b).item()
    return score, path_a, path_b
