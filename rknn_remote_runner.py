import argparse
import json
from pathlib import Path

import numpy as np

from utils.rknn_audio import cosine_score, create_runtime, extract_embedding


def _bool_from_disable_flag(args):
    return not bool(args.disable_inference_preprocess)


def _decision_from_score(score, threshold):
    return "accept" if float(score) >= float(threshold) else "reject"


def _write_json(output_path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _add_common_runner_args(parser):
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the RKNN model.",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=5,
        help="Number of evenly spaced chunks for long audio.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Optional RKNN target platform. Leave empty on-board.",
    )
    parser.add_argument(
        "--disable-inference-preprocess",
        action="store_true",
        help="Disable inference-only preprocessing (silence trimming and loudness normalization).",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remote audio runner for board-side RKNN verification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser(
        "embed-audio",
        help="Generate one normalized embedding from a real audio file.",
    )
    embed_parser.add_argument("--audio", required=True, help="Input audio path.")
    embed_parser.add_argument(
        "--output-embedding",
        required=True,
        help="Output embedding .npy path.",
    )
    _add_common_runner_args(embed_parser)

    verify_parser = subparsers.add_parser(
        "verify-audio",
        help="Compare an audio file against an active profile embedding.",
    )
    verify_parser.add_argument("--audio", required=True, help="Input audio path.")
    verify_parser.add_argument(
        "--profile",
        required=True,
        help="Input active profile embedding .npy path.",
    )
    verify_parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Cosine threshold for accept/reject.",
    )
    verify_parser.add_argument(
        "--output-json",
        required=True,
        help="Output result JSON path.",
    )
    _add_common_runner_args(verify_parser)
    return parser.parse_args()


def run_embed_audio(args):
    audio_path = Path(args.audio)
    model_path = Path(args.model)
    output_path = Path(args.output_embedding)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    rknn_lite = create_runtime(
        model_path,
        verbose=False,
        target=args.target or None,
    )
    try:
        embedding = extract_embedding(
            rknn_lite,
            str(audio_path),
            num_eval=args.num_eval,
            preprocess_for_inference=_bool_from_disable_flag(args),
        )
    finally:
        rknn_lite.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding.astype(np.float32))
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    print(f"Output embedding: {output_path}")
    print(f"Embedding shape: {tuple(embedding.shape)}")


def run_verify_audio(args):
    audio_path = Path(args.audio)
    model_path = Path(args.model)
    profile_path = Path(args.profile)
    output_path = Path(args.output_json)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    profile = np.load(profile_path).astype(np.float32).reshape(1, -1)
    rknn_lite = create_runtime(
        model_path,
        verbose=False,
        target=args.target or None,
    )
    try:
        embedding = extract_embedding(
            rknn_lite,
            str(audio_path),
            num_eval=args.num_eval,
            preprocess_for_inference=_bool_from_disable_flag(args),
        )
    finally:
        rknn_lite.release()

    score = cosine_score(embedding, profile)
    decision = _decision_from_score(score, args.threshold)
    payload = {
        "score": float(score),
        "threshold": float(args.threshold),
        "passed": decision == "accept",
        "decision": decision,
        "preprocess_for_inference": _bool_from_disable_flag(args),
        "num_eval": int(args.num_eval),
    }
    _write_json(output_path, payload)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    print(f"Profile: {profile_path}")
    print(f"Output json: {output_path}")
    print(json.dumps(payload, ensure_ascii=False))


def main():
    args = parse_args()
    if args.command == "embed-audio":
        run_embed_audio(args)
        return
    if args.command == "verify-audio":
        run_verify_audio(args)
        return
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
