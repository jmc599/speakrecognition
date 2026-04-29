import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from export_feature_npy import build_feature
from utils.legacy_feature import DEFAULT_MAX_FRAMES


SUFFIXES = {".wav", ".flac", ".mp3", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a profile evaluation set with precomputed .npy features."
    )
    parser.add_argument(
        "--source-mode",
        type=str,
        default="dataset",
        choices=["dataset", "official_eval"],
        help="Input source mode. Use official_eval for strict held-out profile export.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Dataset root containing speaker folders such as data/id00000/*.flac.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="",
        help="Base directory used to resolve relative paths from official eval trial lists.",
    )
    parser.add_argument(
        "--trials",
        type=str,
        default="",
        help="Official eval trials list path.",
    )
    parser.add_argument(
        "--enroll-list",
        type=str,
        default="",
        help="Official eval enroll list path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rknn_eval_set",
        help="Directory to save exported features and pair manifest.",
    )
    parser.add_argument(
        "--same-pairs",
        type=int,
        default=10,
        help="Number of same-speaker pairs to record in pairs.csv.",
    )
    parser.add_argument(
        "--diff-pairs",
        type=int,
        default=10,
        help="Number of different-speaker pairs to record in pairs.csv.",
    )
    parser.add_argument(
        "--enroll-count",
        type=int,
        default=3,
        help="Enrollment samples required per speaker for strict profile export.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Frame length used when exporting features.",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=5,
        help="Number of evenly spaced chunks for long audio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--for-inference",
        action="store_true",
        help="Apply inference-only preprocessing when exporting features.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.source_mode == "dataset":
        if not args.dataset_root:
            raise ValueError("--dataset-root is required with --source-mode dataset.")
        return

    missing = [
        name
        for name, value in (
            ("--base-path", args.base_path),
            ("--trials", args.trials),
            ("--enroll-list", args.enroll_list),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "official_eval mode requires: " + ", ".join(missing)
        )


def speaker_id_from_relative(relative_path):
    rel_path = Path(relative_path)
    parts = rel_path.parts
    if len(parts) > 1:
        return parts[0]
    return rel_path.parent.name or rel_path.stem or "unknown"


def speaker_id_from_model_id(model_id):
    if model_id.endswith("-enroll"):
        return model_id[: -len("-enroll")]
    return model_id


def speaker_id_from_eval_test(relative_path):
    rel_path = Path(relative_path)
    if rel_path.parent.name == "test":
        stem = rel_path.stem
        if "-" in stem:
            return stem.split("-", 1)[0]
    return speaker_id_from_relative(relative_path)


def feature_name_from_relative(relative_path):
    rel_path = Path(relative_path)
    without_suffix = rel_path.with_suffix("")
    parts = without_suffix.parts or (without_suffix.name,)
    return "__".join(parts) + ".npy"


def add_audio_entry(by_speaker, speaker_id, audio_path, relative_key):
    by_speaker[speaker_id][str(relative_key)] = {
        "speaker_id": speaker_id,
        "audio_path": Path(audio_path),
        "relative_key": str(relative_key),
    }


def collect_audio_dataset(dataset_root):
    dataset_root = Path(dataset_root)
    by_speaker = defaultdict(dict)
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        speaker_id = path.parent.name
        relative_key = path.relative_to(dataset_root)
        add_audio_entry(by_speaker, speaker_id, path, relative_key)
    total_unique_audio = sum(len(items) for items in by_speaker.values())
    return finalize_audio_groups(by_speaker), total_unique_audio


def collect_audio_official_eval(base_path, trials_path, enroll_list_path):
    base_path = Path(base_path)
    dataset_root = base_path.parent
    enroll_map_path = Path(enroll_list_path).with_name("enroll.map")
    if not enroll_map_path.is_file():
        raise FileNotFoundError(f"enroll.map not found: {enroll_map_path}")

    by_speaker = defaultdict(dict)
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            test_relative = parts[1]
            resolved = resolve_audio_with_suffix_fallback(base_path, test_relative)
            speaker_id = speaker_id_from_eval_test(test_relative)
            add_audio_entry(by_speaker, speaker_id, resolved, test_relative)

    with enroll_map_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            speaker_id = speaker_id_from_model_id(parts[0])
            for source_relative in parts[1:]:
                resolved = resolve_official_source_audio(dataset_root, source_relative)
                relative_key = str(resolved.relative_to(dataset_root))
                add_audio_entry(by_speaker, speaker_id, resolved, relative_key)

    total_unique_audio = sum(len(items) for items in by_speaker.values())
    return finalize_audio_groups(by_speaker), total_unique_audio


def resolve_official_source_audio(dataset_root, source_relative):
    source_relative = Path(source_relative)
    candidates = [dataset_root / "data" / source_relative, dataset_root / source_relative]
    return resolve_first_existing(candidates, source_relative)


def resolve_audio_with_suffix_fallback(base_dir, relative_path):
    relative_path = Path(relative_path)
    candidates = [Path(base_dir) / relative_path]
    return resolve_first_existing(candidates, relative_path)


def resolve_first_existing(candidates, relative_path):
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for candidate in candidates:
        stem = candidate.with_suffix("")
        for suffix in (".flac", ".wav", ".mp3", ".m4a"):
            patched = stem.with_suffix(suffix)
            if patched.is_file():
                return patched

    raise FileNotFoundError(
        f"Audio file not found for relative path: {relative_path}"
    )


def finalize_audio_groups(by_speaker):
    finalized = {}
    for speaker_id, items in by_speaker.items():
        finalized[speaker_id] = sorted(
            items.values(),
            key=lambda item: item["relative_key"],
        )
    return finalized


def filter_speakers(by_speaker, min_files_per_speaker):
    usable = {
        speaker_id: entries
        for speaker_id, entries in by_speaker.items()
        if len(entries) >= min_files_per_speaker
    }
    excluded = sorted(
        speaker_id
        for speaker_id, entries in by_speaker.items()
        if len(entries) < min_files_per_speaker
    )
    return usable, excluded


def sample_same_pairs(by_speaker, count, rng):
    candidates = []
    for speaker_id, entries in by_speaker.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                candidates.append((speaker_id, entries[i], entries[j], 1))
    rng.shuffle(candidates)
    return candidates[:count]


def sample_diff_pairs(by_speaker, count, rng):
    speaker_ids = [speaker_id for speaker_id, entries in by_speaker.items() if entries]
    pairs = []
    seen = set()
    max_trials = max(count * 50, 100)
    for _ in range(max_trials):
        if len(pairs) >= count:
            break
        speaker_a, speaker_b = rng.sample(speaker_ids, 2)
        entry_a = rng.choice(by_speaker[speaker_a])
        entry_b = rng.choice(by_speaker[speaker_b])
        pair_key = (
            tuple(sorted((speaker_a, speaker_b))),
            entry_a["relative_key"],
            entry_b["relative_key"],
        )
        if pair_key in seen:
            continue
        seen.add(pair_key)
        pairs.append((f"{speaker_a}|{speaker_b}", entry_a, entry_b, 0))
    return pairs


def export_feature(entry, out_dir, feature_map, *, max_frames, num_eval, for_inference):
    relative_key = entry["relative_key"]
    if relative_key in feature_map:
        return feature_map[relative_key]

    speaker_dir = out_dir / entry["speaker_id"]
    speaker_dir.mkdir(parents=True, exist_ok=True)
    out_path = speaker_dir / feature_name_from_relative(relative_key)
    if not out_path.is_file():
        feature = build_feature(
            str(entry["audio_path"]),
            max_frames=max_frames,
            num_eval=num_eval,
            for_inference=for_inference,
        )
        np.save(out_path, feature.astype(np.float32))
    feature_map[relative_key] = out_path
    return out_path


def export_all_features(by_speaker, out_dir, *, max_frames, num_eval, for_inference):
    feature_map = {}
    all_entries = [
        entry
        for entries in by_speaker.values()
        for entry in entries
    ]
    for entry in tqdm(all_entries, desc="Export features"):
        export_feature(
            entry,
            out_dir,
            feature_map,
            max_frames=max_frames,
            num_eval=num_eval,
            for_inference=for_inference,
        )
    return feature_map


def write_manifest(output_dir, pairs, feature_map):
    manifest_path = output_dir / "pairs.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "speaker_key",
                "audio_a",
                "audio_b",
                "feature_a",
                "feature_b",
            ]
        )
        for speaker_key, entry_a, entry_b, label in pairs:
            writer.writerow(
                [
                    label,
                    speaker_key,
                    str(entry_a["audio_path"]),
                    str(entry_b["audio_path"]),
                    str(feature_map[entry_a["relative_key"]].relative_to(output_dir)),
                    str(feature_map[entry_b["relative_key"]].relative_to(output_dir)),
                ]
            )
    return manifest_path


def write_metadata(
    output_dir,
    *,
    args,
    num_unique_audio,
    num_speakers_total,
    num_speakers_usable,
    excluded_speakers,
    num_features_exported,
):
    metadata_path = output_dir / "metadata.json"
    payload = {
        "source_mode": args.source_mode,
        "dataset_root": args.dataset_root,
        "base_path": args.base_path,
        "trials": args.trials,
        "enroll_list": args.enroll_list,
        "num_unique_audio": num_unique_audio,
        "num_speakers_total": num_speakers_total,
        "num_speakers_usable": num_speakers_usable,
        "num_speakers_excluded": len(excluded_speakers),
        "excluded_speakers": excluded_speakers,
        "min_files_per_speaker": args.enroll_count + 1,
        "same_pairs": args.same_pairs,
        "diff_pairs": args.diff_pairs,
        "seed": args.seed,
        "max_frames": args.max_frames,
        "num_eval": args.num_eval,
        "preprocess_for_inference": bool(args.for_inference),
        "num_features_exported": num_features_exported,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return metadata_path


def main():
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    if args.source_mode == "official_eval":
        by_speaker_all, num_unique_audio = collect_audio_official_eval(
            args.base_path,
            args.trials,
            args.enroll_list,
        )
    else:
        by_speaker_all, num_unique_audio = collect_audio_dataset(args.dataset_root)

    if not by_speaker_all:
        raise RuntimeError("No audio files found for the selected source mode.")

    by_speaker, excluded_speakers = filter_speakers(
        by_speaker_all,
        min_files_per_speaker=args.enroll_count + 1,
    )
    if not by_speaker:
        raise RuntimeError(
            "No speakers have enough audio for strict profile export. "
            f"Need at least {args.enroll_count + 1} files per speaker."
        )

    feature_map = export_all_features(
        by_speaker,
        feature_dir,
        max_frames=args.max_frames,
        num_eval=args.num_eval,
        for_inference=args.for_inference,
    )

    same_pairs = sample_same_pairs(by_speaker, args.same_pairs, rng)
    diff_pairs = sample_diff_pairs(by_speaker, args.diff_pairs, rng)
    pairs = same_pairs + diff_pairs
    rng.shuffle(pairs)

    manifest_path = write_manifest(output_dir, pairs, feature_map)
    metadata_path = write_metadata(
        output_dir,
        args=args,
        num_unique_audio=num_unique_audio,
        num_speakers_total=len(by_speaker_all),
        num_speakers_usable=len(by_speaker),
        excluded_speakers=excluded_speakers,
        num_features_exported=len(feature_map),
    )

    print(f"Source mode: {args.source_mode}")
    if args.source_mode == "official_eval":
        print(f"Base path: {args.base_path}")
        print(f"Trials: {args.trials}")
        print(f"Enroll list: {args.enroll_list}")
    else:
        print(f"Dataset root: {args.dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Feature dir: {feature_dir}")
    print(f"Unique audio discovered: {num_unique_audio}")
    print(f"Usable speakers: {len(by_speaker)}/{len(by_speaker_all)}")
    print(f"Features exported: {len(feature_map)}")
    print(f"Pairs saved: {len(pairs)}")
    print(f"Same pairs: {sum(1 for _, _, _, label in pairs if label == 1)}")
    print(f"Diff pairs: {sum(1 for _, _, _, label in pairs if label == 0)}")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Inference preprocess: {args.for_inference}")


if __name__ == "__main__":
    main()
