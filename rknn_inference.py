import argparse
from pathlib import Path

from utils.rknn_audio import cosine_score, create_runtime, extract_embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run speaker verification with an RKNN model on RK3588."
    )
    parser.add_argument("audio_a", type=str, help="First audio file path.")
    parser.add_argument("audio_b", type=str, help="Second audio file path.")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/resnet_v7_vox2ft_s1_latest_fixed_rt160_nonorm.rknn",
        help="Path to the RKNN model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional cosine threshold for same-speaker decision. Leave unset to print score only.",
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
        default="rk3588",
        help="RKNN target platform.",
    )
    parser.add_argument(
        "--disable-inference-preprocess",
        action="store_true",
        help="Disable inference-only preprocessing (silence trimming and loudness normalization).",
    )
    return parser.parse_args()
def main():
    args = parse_args()
    model_path = Path(args.model)

    print("--> Load RKNN model")
    rknn_lite = create_runtime(
        model_path,
        verbose=True,
        target=args.target,
    )
    print("done")

    emb_a = extract_embedding(
        rknn_lite,
        args.audio_a,
        num_eval=args.num_eval,
        preprocess_for_inference=not args.disable_inference_preprocess,
    )
    emb_b = extract_embedding(
        rknn_lite,
        args.audio_b,
        num_eval=args.num_eval,
        preprocess_for_inference=not args.disable_inference_preprocess,
    )
    score = cosine_score(emb_a, emb_b)
    verdict = (
        "same speaker" if args.threshold is not None and score >= args.threshold
        else "different speaker" if args.threshold is not None
        else "threshold unset"
    )

    print(f"Model: {model_path}")
    print(f"Audio A: {args.audio_a}")
    print(f"Audio B: {args.audio_b}")
    print(f"Cosine score: {score:.6f}")
    if args.threshold is None:
        print("Threshold: <unset>")
    else:
        print(f"Threshold: {args.threshold:.6f}")
    print(f"Inference preprocess: {not args.disable_inference_preprocess}")
    print(f"Decision: {verdict}")

    rknn_lite.release()


if __name__ == "__main__":
    main()
