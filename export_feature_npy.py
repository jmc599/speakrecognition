import argparse
from pathlib import Path

import numpy as np
import torch

from utils.legacy_feature import DEFAULT_MAX_FRAMES
from utils.audio_io import load_audio_mono_16k
from utils.speaker_verification import build_mel_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a fixed-shape log-mel feature tensor to .npy."
    )
    parser.add_argument("audio", type=str, help="Input audio path.")
    parser.add_argument(
        "--output",
        type=str,
        default="feature.npy",
        help="Output .npy path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Target frame length.",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=5,
        help="Number of evenly spaced chunks for long audio. Matches the original verification pipeline.",
    )
    parser.add_argument(
        "--for-inference",
        action="store_true",
        help="Apply inference-only preprocessing: trim front/back silence and normalize loudness.",
    )
    return parser.parse_args()


def build_feature(
    audio_path,
    max_frames=DEFAULT_MAX_FRAMES,
    num_eval=5,
    for_inference=False,
):
    waveform, _ = load_audio_mono_16k(audio_path, for_inference=for_inference)
    mel_transform = build_mel_transform()

    mel_spec = mel_transform(waveform)
    logmel = np.log((mel_spec.numpy()) + 1e-6).astype(np.float32)

    total_frames = logmel.shape[2]
    if total_frames < max_frames:
        pad = max_frames - total_frames
        logmel = np.pad(logmel, ((0, 0), (0, 0), (0, pad)), mode="constant")
        return logmel[None, ...]

    if total_frames == max_frames:
        return logmel[None, ...]

    if num_eval <= 1:
        starts = [0]
    else:
        max_start = total_frames - max_frames
        starts = torch.linspace(0, max_start, steps=num_eval).long().tolist()

    chunks = [logmel[:, :, start : start + max_frames] for start in starts]
    return np.stack(chunks, axis=0).astype(np.float32)


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    output_path = Path(args.output)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    feature = build_feature(
        str(audio_path),
        max_frames=args.max_frames,
        num_eval=args.num_eval,
        for_inference=args.for_inference,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature.astype(np.float32))

    print(f"Audio: {audio_path}")
    print(f"Output: {output_path}")
    print(f"Feature shape: {feature.shape}")
    print(f"Feature dtype: {feature.dtype}")
    print(f"Inference preprocess: {args.for_inference}")


if __name__ == "__main__":
    main()
