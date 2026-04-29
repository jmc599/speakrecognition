import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrate_profile_results import default_output_path
from gui.services.audio_pipeline import normalize_audio_file
from gui.services.board_client import BoardClient
from gui.services.config_store import ConfigStore, THRESHOLD_POLICY_CHOICES
from utils.detection_metrics import evaluate_fixed_threshold, load_score_label_pairs
from utils.legacy_feature import DEFAULT_MAX_FRAMES


REQUIRED_COLUMNS = ("trial_id", "label", "enroll_path", "verify_path", "notes")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Review remote_rknn threshold on a small manually labeled phone-import "
            "trial set without changing the GUI workflow."
        )
    )
    parser.add_argument(
        "--trials",
        required=True,
        help="CSV path with columns: trial_id,label,enroll_path,verify_path,notes",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory for normalized audio, cached enroll embeddings, "
            "results CSV, calibration JSON, and report."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Current threshold to review. Defaults to the configured remote_rknn threshold.",
    )
    parser.add_argument(
        "--threshold-policy",
        default="far_1",
        choices=THRESHOLD_POLICY_CHOICES,
        help="Threshold policy recorded in the calibration output.",
    )
    return parser.parse_args()


def timestamp_slug():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_review_dir():
    return Path("artifacts") / "results" / f"remote_rknn_threshold_review_{timestamp_slug()}"


def sha1_text(value):
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()


def slugify(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._-") or "trial"


def resolve_audio_path(base_dir, value):
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def load_trials(trials_path):
    with trials_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Trials CSV is empty: {trials_path}")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Trials CSV missing required columns: {', '.join(missing)}"
            )

        rows = []
        seen_ids = set()
        base_dir = trials_path.parent.resolve()
        for index, raw in enumerate(reader, start=1):
            trial_id = str(raw["trial_id"]).strip()
            if not trial_id:
                raise ValueError(f"Row {index} has empty trial_id.")
            if trial_id in seen_ids:
                raise ValueError(f"Duplicate trial_id detected: {trial_id}")
            seen_ids.add(trial_id)

            label_text = str(raw["label"]).strip()
            if label_text not in {"0", "1"}:
                raise ValueError(
                    f"Row {index} has invalid label {label_text!r}; expected 0 or 1."
                )

            rows.append(
                {
                    "trial_id": trial_id,
                    "label": int(label_text),
                    "enroll_path": resolve_audio_path(base_dir, raw["enroll_path"]),
                    "verify_path": resolve_audio_path(base_dir, raw["verify_path"]),
                    "notes": str(raw.get("notes", "") or "").strip(),
                }
            )

    if not rows:
        raise ValueError(f"No trials found in CSV: {trials_path}")
    return rows


def normalize_cached(source_path, cache_dir, prefix, cache_index):
    cache_key = str(source_path.resolve()).lower()
    if cache_key in cache_index:
        return cache_index[cache_key]

    temp_output = Path(normalize_audio_file(str(source_path), prefix=prefix))
    cache_path = cache_dir / f"{sha1_text(cache_key)}.wav"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_output), str(cache_path))
    cache_index[cache_key] = cache_path
    return cache_path


def load_or_build_enroll_embedding(board, normalized_audio_path, cache_dir):
    cache_key = str(normalized_audio_path.resolve()).lower()
    cache_path = cache_dir / f"{sha1_text(cache_key)}.npy"
    if cache_path.is_file():
        return cache_path

    result = board.embed_audio(str(normalized_audio_path), prefix="threshold_enroll")
    embedding = np.asarray(result["embedding"], dtype=np.float32).reshape(1, -1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding.astype(np.float32))
    return cache_path


def write_results_csv(rows, results_path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "label",
                "enroll_path",
                "verify_path",
                "score",
                "threshold",
                "decision",
                "passed",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_calibration(results_path, calibration_path, config, threshold_policy):
    command = [
        sys.executable,
        "calibrate_profile_results.py",
        "--results",
        str(results_path),
        "--output",
        str(calibration_path),
        "--checkpoint",
        str(config["remote_model_path"]),
        "--max-frames",
        str(DEFAULT_MAX_FRAMES),
        "--preprocess-for-inference",
        "--backend-mode",
        "remote_rknn",
        "--threshold-policy",
        str(threshold_policy),
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Calibration command failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    with calibration_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_misclassification_rows(result_rows, threshold):
    current_misclassified = []
    for row in result_rows:
        label = int(row["label"])
        score = float(row["score"])
        if (label == 1 and score < threshold) or (label == 0 and score >= threshold):
            current_misclassified.append(row)
    return current_misclassified


def write_report(report_path, result_rows, threshold, fixed_metrics, calibration_summary):
    scores = np.asarray([float(row["score"]) for row in result_rows], dtype=np.float32)
    labels = np.asarray([int(row["label"]) for row in result_rows], dtype=np.int32)
    recommended_threshold = float(calibration_summary["selected_threshold"])
    recommended_metrics = evaluate_fixed_threshold(scores, labels, recommended_threshold)
    misclassified_rows = build_misclassification_rows(result_rows, threshold)
    same_count = int(labels.sum())
    diff_count = int(len(labels) - same_count)

    lines = [
        "# Remote RKNN 阈值复核报告",
        "",
        f"- 样本总数: {len(result_rows)}",
        f"- 同人样本数: {same_count}",
        f"- 异人样本数: {diff_count}",
        f"- 当前阈值: {threshold:.6f}",
        f"- 当前阈值准确率: {fixed_metrics['accuracy'] * 100:.2f}%",
        f"- 当前阈值 FAR: {fixed_metrics['far'] * 100:.2f}%",
        f"- 当前阈值 FRR: {fixed_metrics['frr'] * 100:.2f}%",
        "",
        "## 拟合结果",
        "",
        f"- EER: {calibration_summary['eer'] * 100:.4f}%",
        f"- EER threshold: {float(calibration_summary['eer_threshold']):.6f}",
        f"- FAR<=1% 推荐阈值: {float(calibration_summary['threshold_far_1']):.6f}",
        f"- FAR<=0.1% 推荐阈值: {float(calibration_summary['threshold_far_0p1']):.6f}",
        f"- 当前 policy ({calibration_summary['threshold_policy']}) 选中阈值: {recommended_threshold:.6f}",
        f"- 推荐阈值准确率: {recommended_metrics['accuracy'] * 100:.2f}%",
        f"- 推荐阈值 FAR: {recommended_metrics['far'] * 100:.2f}%",
        f"- 推荐阈值 FRR: {recommended_metrics['frr'] * 100:.2f}%",
        "",
        "## 当前阈值下的明显误判样例",
        "",
    ]

    if not misclassified_rows:
        lines.append("- 无")
    else:
        for row in misclassified_rows:
            lines.append(
                "- "
                f"{row['trial_id']}: label={row['label']}, score={float(row['score']):.6f}, "
                f"decision={row['decision']}, "
                f"enroll={row['enroll_path']}, verify={row['verify_path']}"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    trials_path = Path(args.trials).resolve()
    if not trials_path.is_file():
        raise FileNotFoundError(f"Trials CSV not found: {trials_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_review_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = output_dir / "normalized_audio"
    enroll_cache_dir = output_dir / "enroll_embedding_cache"
    raw_results_dir = output_dir / "raw_results"
    results_path = output_dir / "results.csv"
    calibration_path = default_output_path(results_path)
    report_path = output_dir / "report.md"

    config = ConfigStore().load()
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(config.get("remote_rknn_threshold") or 0.342024)
    )
    trials = load_trials(trials_path)

    board = BoardClient(config)
    board.test_connection()

    normalized_cache = {}
    result_rows = []
    run_id = timestamp_slug()
    remote_review_root = f"{config['remote_workdir'].rstrip('/')}/threshold_review/{run_id}"

    try:
        for row in trials:
            trial_slug = slugify(row["trial_id"])
            remote_trial_dir = f"{remote_review_root}/{trial_slug}"
            remote_profile = f"{remote_trial_dir}/profile.npy"
            raw_json_path = raw_results_dir / f"{trial_slug}.json"

            enroll_audio = normalize_cached(
                row["enroll_path"],
                normalized_dir,
                prefix="threshold_enroll",
                cache_index=normalized_cache,
            )
            verify_audio = normalize_cached(
                row["verify_path"],
                normalized_dir,
                prefix="threshold_verify",
                cache_index=normalized_cache,
            )
            enroll_embedding_path = load_or_build_enroll_embedding(
                board,
                enroll_audio,
                enroll_cache_dir,
            )

            try:
                board.ensure_remote_dir(remote_trial_dir)
                board.upload_file(enroll_embedding_path, remote_profile)
                verify_result = board.verify_audio(
                    str(verify_audio),
                    threshold=threshold,
                    prefix=f"threshold_review_{trial_slug}",
                    profile_path=remote_profile,
                    remote_dir=remote_trial_dir,
                )
            finally:
                try:
                    board.remove_remote_path(remote_trial_dir, recursive=True)
                except Exception:
                    pass

            raw_json_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(verify_result["result_path"], raw_json_path)

            result_row = {
                "trial_id": row["trial_id"],
                "label": int(row["label"]),
                "enroll_path": str(row["enroll_path"]),
                "verify_path": str(row["verify_path"]),
                "score": f"{float(verify_result['score']):.9f}",
                "threshold": f"{float(verify_result['threshold']):.6f}",
                "decision": str(verify_result["decision"]),
                "passed": int(bool(verify_result["passed"])),
                "notes": row["notes"],
            }
            result_rows.append(result_row)
            write_results_csv(result_rows, results_path)
    finally:
        try:
            board.remove_remote_path(remote_review_root, recursive=True)
        except Exception:
            pass
        board.close()

    calibration_summary = run_calibration(
        results_path,
        calibration_path,
        config=config,
        threshold_policy=args.threshold_policy,
    )
    scores, labels = load_score_label_pairs(results_path)
    fixed_metrics = evaluate_fixed_threshold(scores, labels, threshold)
    write_report(
        report_path,
        result_rows=result_rows,
        threshold=threshold,
        fixed_metrics=fixed_metrics,
        calibration_summary=calibration_summary,
    )

    print(f"Trials CSV: {trials_path}")
    print(f"Output dir: {output_dir}")
    print(f"Results CSV: {results_path}")
    print(f"Calibration JSON: {calibration_path}")
    print(f"Report: {report_path}")
    print(f"Current threshold: {threshold:.6f}")
    print(f"Trials processed: {len(result_rows)}")


if __name__ == "__main__":
    main()
