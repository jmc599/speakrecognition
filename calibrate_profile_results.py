import argparse
import json
from pathlib import Path

from utils.detection_metrics import (
    THRESHOLD_POLICY_CHOICES,
    build_calibration_summary,
    build_threshold_transfer_summary,
    load_score_label_pairs,
)
from utils.legacy_feature import DEFAULT_MAX_FRAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate profile thresholds from a results CSV."
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="CSV path with label and score columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional calibration JSON path. Defaults to <results>_calibration.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint or model identifier recorded in the calibration output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Frame length used to produce the scores.",
    )
    parser.add_argument(
        "--preprocess-for-inference",
        action="store_true",
        help="Record that inference preprocessing was enabled when producing the scores.",
    )
    parser.add_argument(
        "--backend-mode",
        type=str,
        default="",
        choices=["", "local_onnx", "local_pytorch", "remote_rknn", "wespeaker_onnx"],
        help="Optional GUI backend key to update when --write-gui-config is set.",
    )
    parser.add_argument(
        "--reference-calibration",
        type=str,
        default="",
        help=(
            "Optional calibration JSON to reuse thresholds from. When provided, this "
            "command evaluates the current results with those fixed thresholds instead "
            "of fitting thresholds on the current results."
        ),
    )
    parser.add_argument(
        "--write-gui-config",
        action="store_true",
        help="Write the calibrated threshold back to the GUI config for the selected backend.",
    )
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="eer",
        choices=THRESHOLD_POLICY_CHOICES,
        help="Operating point policy used for selected_threshold and GUI writes.",
    )
    return parser.parse_args()


def default_output_path(results_path):
    stem = results_path.stem
    if stem.endswith(".csv"):
        stem = stem[:-4]
    return results_path.with_name(f"{stem}_calibration.json")


def maybe_write_gui_config(summary, backend_mode):
    if not backend_mode:
        raise ValueError("--backend-mode is required with --write-gui-config.")

    from gui.services.config_store import (
        ConfigStore,
        THRESHOLD_SOURCE_CALIBRATION,
        backend_threshold_key,
        set_backend_threshold_metadata,
    )

    threshold_key = backend_threshold_key(backend_mode)
    selected_threshold = float(summary["selected_threshold"])

    store = ConfigStore()
    config = store.load()
    set_backend_threshold_metadata(
        config,
        backend_mode,
        threshold=selected_threshold,
        threshold_policy=summary["threshold_policy"],
        source=THRESHOLD_SOURCE_CALIBRATION,
        calibrated=True,
    )
    if backend_mode == "local_pytorch":
        config["backend_mode"] = "local_pytorch"
        if summary.get("checkpoint"):
            config["local_pytorch_checkpoint_path"] = str(summary["checkpoint"])
    if backend_mode == "local_onnx":
        config["backend_mode"] = "local_onnx"
        if summary.get("checkpoint"):
            config["local_onnx_model_path"] = str(summary["checkpoint"])
    if backend_mode == "remote_rknn":
        config["backend_mode"] = "remote_rknn"
        if summary.get("checkpoint"):
            config["remote_model_path"] = str(summary["checkpoint"])
    if backend_mode == "wespeaker_onnx":
        config["backend_mode"] = "wespeaker_onnx"
        if summary.get("checkpoint"):
            config["wespeaker_onnx_model_path"] = str(summary["checkpoint"])
        config["wespeaker_threshold_calibrated"] = True
    store.save(config)
    return threshold_key


def main():
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_path = Path(args.output) if args.output else default_output_path(results_path)
    scores, labels = load_score_label_pairs(results_path)
    if args.reference_calibration:
        reference_path = Path(args.reference_calibration)
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Reference calibration not found: {reference_path}"
            )
        with reference_path.open("r", encoding="utf-8") as f:
            reference_summary = json.load(f)
        summary = build_threshold_transfer_summary(
            scores,
            labels,
            reference_summary,
            checkpoint=args.checkpoint,
            max_frames=args.max_frames,
            preprocess_for_inference=args.preprocess_for_inference,
            threshold_policy=args.threshold_policy,
        )
        summary["reference_calibration"] = str(reference_path)
    else:
        summary = build_calibration_summary(
            scores,
            labels,
            checkpoint=args.checkpoint,
            max_frames=args.max_frames,
            preprocess_for_inference=args.preprocess_for_inference,
            threshold_policy=args.threshold_policy,
        )
    summary["results"] = str(results_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Results: {results_path}")
    print(f"Calibration: {output_path}")
    print(f"Trials: {summary['num_trials']}")
    if "eer" in summary:
        print(f"EER: {summary['eer'] * 100:.4f}%")
        print(f"EER threshold: {summary['eer_threshold']:.6f}")
    if "min_dcf_p01" in summary:
        print(f"minDCF@0.01: {summary['min_dcf_p01']:.6f}")
    print(f"Threshold policy: {summary['threshold_policy']}")
    print(f"Selected threshold: {summary['selected_threshold']:.6f}")
    print(f"Selected FAR: {summary['selected_far'] * 100:.4f}%")
    print(f"Selected FRR: {summary['selected_frr'] * 100:.4f}%")
    print(
        f"FRR@FAR<=1%: {summary['frr_at_far_1'] * 100:.4f}% "
        f"(threshold={summary['threshold_far_1']:.6f})"
    )
    print(
        f"FRR@FAR<=0.1%: {summary['frr_at_far_0p1'] * 100:.4f}% "
        f"(threshold={summary['threshold_far_0p1']:.6f})"
    )

    if args.write_gui_config:
        threshold_key = maybe_write_gui_config(summary, args.backend_mode)
        print(
            f"GUI config updated: {threshold_key}={summary['selected_threshold']:.6f}"
        )


if __name__ == "__main__":
    main()
