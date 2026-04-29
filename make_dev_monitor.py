from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


SPK_PATTERN = re.compile(r"id\d+", flags=re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create mutually exclusive main-train / dev-monitor / dev-cal splits from train_list_open_set.txt."
    )
    parser.add_argument("--train-list", type=str, required=True)
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--monitor-speaker-ratio", type=float, default=0.05)
    parser.add_argument("--cal-speaker-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="lists/dev_splits")
    parser.add_argument("--monitor-target-trials", type=int, default=1000)
    parser.add_argument("--monitor-nontarget-trials", type=int, default=1000)
    parser.add_argument("--cal-target-trials", type=int, default=1000)
    parser.add_argument("--cal-nontarget-trials", type=int, default=1000)
    return parser.parse_args()


def infer_speaker_key(rel_path: str) -> str:
    normalized = rel_path.replace("\\", "/")
    match = SPK_PATTERN.search(normalized)
    if match:
        return match.group(0).lower()
    raise ValueError(f"Could not infer speaker key from path: {rel_path}")


def load_train_entries(train_list_path: Path):
    entries = []
    for line in train_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rel_path, label = line.split()
        rel_path = rel_path.replace("\\", "/")
        speaker_key = infer_speaker_key(rel_path)
        entries.append(
            {
                "rel_path": rel_path,
                "speaker_key": speaker_key,
                "orig_label": int(label),
            }
        )
    entries.sort(key=lambda item: (item["speaker_key"], item["rel_path"], item["orig_label"]))
    return entries


def group_by_speaker(entries):
    by_speaker = defaultdict(list)
    for entry in entries:
        by_speaker[entry["speaker_key"]].append(entry["rel_path"])
    return {key: sorted(paths) for key, paths in sorted(by_speaker.items())}


def choose_split_speakers(eligible_speakers, monitor_ratio, cal_ratio, rng):
    speakers = list(eligible_speakers)
    rng.shuffle(speakers)

    monitor_count = max(1, int(round(len(speakers) * float(monitor_ratio))))
    cal_count = max(1, int(round(len(speakers) * float(cal_ratio))))
    if monitor_count + cal_count >= len(speakers):
        raise RuntimeError(
            "Not enough eligible speakers to allocate monitor/cal splits without "
            "overlapping the main train split."
        )

    monitor_speakers = sorted(speakers[:monitor_count])
    cal_speakers = sorted(speakers[monitor_count : monitor_count + cal_count])
    return monitor_speakers, cal_speakers


def allocate_quotas(keys, total):
    if total <= 0 or not keys:
        return {}
    base = total // len(keys)
    remainder = total % len(keys)
    quotas = {}
    for idx, key in enumerate(keys):
        quotas[key] = base + (1 if idx < remainder else 0)
    return quotas


def sample_with_cycles(candidates, count, rng):
    if count <= 0:
        return []
    if not candidates:
        raise RuntimeError("Cannot sample from an empty candidate pool.")

    candidates = list(candidates)
    result = []
    while len(result) < count:
        cycle = list(candidates)
        rng.shuffle(cycle)
        take = min(count - len(result), len(cycle))
        result.extend(cycle[:take])
    return result


def build_split_artifacts(split_name, speaker_keys, by_speaker, output_dir, rng, target_trials, non_target_trials):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    enroll_lines = []
    trials_lines = []
    manifest = {
        "split_name": split_name,
        "num_speakers": len(speaker_keys),
        "speaker_keys": list(speaker_keys),
        "target_trials": int(target_trials),
        "non_target_trials": int(non_target_trials),
        "speakers": {},
    }

    enroll_key_by_speaker = {}
    positives_by_key = {}
    negatives_by_key = {}
    enroll_audio_by_speaker = {}

    for speaker_key in speaker_keys:
        paths = list(by_speaker[speaker_key])
        if len(paths) < 3:
            raise RuntimeError(
                f"Speaker {speaker_key} has fewer than 3 paths and cannot be used for {split_name}."
            )
        enroll_path = paths[0]
        verify_paths = paths[1:]
        enroll_key = f"{speaker_key}-enroll"
        enroll_key_by_speaker[speaker_key] = enroll_key
        enroll_audio_by_speaker[speaker_key] = enroll_path
        positives_by_key[enroll_key] = verify_paths
        negatives = []
        for other_key in speaker_keys:
            if other_key == speaker_key:
                continue
            negatives.extend(by_speaker[other_key][1:])
        negatives_by_key[enroll_key] = negatives
        enroll_lines.append(f"{enroll_key} {enroll_path}")
        manifest["speakers"][speaker_key] = {
            "enroll_key": enroll_key,
            "enroll_path": enroll_path,
            "num_audio": len(paths),
            "num_verify_audio": len(verify_paths),
        }

    enroll_keys = [enroll_key_by_speaker[speaker] for speaker in speaker_keys]
    target_quotas = allocate_quotas(enroll_keys, target_trials)
    non_target_quotas = allocate_quotas(enroll_keys, non_target_trials)

    for enroll_key in enroll_keys:
        for rel_path in sample_with_cycles(positives_by_key[enroll_key], target_quotas.get(enroll_key, 0), rng):
            trials_lines.append(f"{enroll_key} {rel_path} 1")
        for rel_path in sample_with_cycles(negatives_by_key[enroll_key], non_target_quotas.get(enroll_key, 0), rng):
            trials_lines.append(f"{enroll_key} {rel_path} 0")

    rng.shuffle(trials_lines)
    (split_dir / "enroll.lst").write_text("\n".join(enroll_lines) + "\n", encoding="utf-8")
    (split_dir / "trials.lst").write_text("\n".join(trials_lines) + "\n", encoding="utf-8")
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return split_dir


def write_train_list_main(entries, excluded_speakers, output_path):
    kept = [entry for entry in entries if entry["speaker_key"] not in excluded_speakers]
    speaker_keys = sorted({entry["speaker_key"] for entry in kept})
    speaker_to_new_label = {speaker: idx for idx, speaker in enumerate(speaker_keys)}

    lines = [
        f"{entry['rel_path']} {speaker_to_new_label[entry['speaker_key']]}"
        for entry in kept
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "num_samples": len(kept),
        "num_speakers": len(speaker_keys),
    }


def validate_audio_paths(entries, base_path: Path):
    missing = []
    for entry in entries:
        if not (base_path / entry["rel_path"]).is_file():
            missing.append(entry["rel_path"])
            if len(missing) >= 20:
                break
    if missing:
        raise FileNotFoundError(
            "Some train-list paths do not exist under base-path. First missing examples: "
            + ", ".join(missing[:5])
        )


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    train_list_path = Path(args.train_list)
    base_path = Path(args.base_path)
    output_dir = Path(args.output_dir)

    if not train_list_path.is_file():
        raise FileNotFoundError(f"Train list not found: {train_list_path}")
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    entries = load_train_entries(train_list_path)
    validate_audio_paths(entries, base_path)
    by_speaker = group_by_speaker(entries)
    eligible_speakers = sorted(
        speaker_key
        for speaker_key, paths in by_speaker.items()
        if len(paths) >= 3
    )
    if len(eligible_speakers) < 10:
        raise RuntimeError("Not enough eligible speakers to build dev-monitor/dev-cal splits.")

    monitor_speakers, cal_speakers = choose_split_speakers(
        eligible_speakers,
        args.monitor_speaker_ratio,
        args.cal_speaker_ratio,
        rng,
    )
    excluded_speakers = set(monitor_speakers) | set(cal_speakers)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = write_train_list_main(
        entries,
        excluded_speakers,
        output_dir / "train_list_main.txt",
    )
    monitor_dir = build_split_artifacts(
        "monitor",
        monitor_speakers,
        by_speaker,
        output_dir,
        random.Random(args.seed + 1),
        args.monitor_target_trials,
        args.monitor_nontarget_trials,
    )
    cal_dir = build_split_artifacts(
        "cal",
        cal_speakers,
        by_speaker,
        output_dir,
        random.Random(args.seed + 2),
        args.cal_target_trials,
        args.cal_nontarget_trials,
    )

    summary = {
        "seed": int(args.seed),
        "train_list": str(train_list_path.resolve()),
        "base_path": str(base_path.resolve()),
        "eligible_speakers": len(eligible_speakers),
        "monitor_speakers": len(monitor_speakers),
        "cal_speakers": len(cal_speakers),
        "monitor_speaker_keys": monitor_speakers,
        "cal_speaker_keys": cal_speakers,
        "train_main_num_samples": train_stats["num_samples"],
        "train_main_num_speakers": train_stats["num_speakers"],
        "monitor_dir": str(monitor_dir.resolve()),
        "cal_dir": str(cal_dir.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Eligible speakers: {len(eligible_speakers)}")
    print(f"Monitor speakers: {len(monitor_speakers)}")
    print(f"Calibration speakers: {len(cal_speakers)}")
    print(f"Train main speakers: {train_stats['num_speakers']}")
    print(f"Train main samples: {train_stats['num_samples']}")
    print(f"Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
