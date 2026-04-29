from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


SPK_PATTERN = re.compile(r"id\d+", flags=re.IGNORECASE)
_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed VoxCeleb2 test monitor split for Stage 1 development "
            "monitoring. This monitor is speaker-disjoint from the Stage 1 train list "
            "and must not be used as a final benchmark."
        )
    )
    parser.add_argument("--base-path", type=str, required=True, help="Dataset root used for scoring.")
    parser.add_argument(
        "--monitor-dir",
        type=str,
        required=True,
        help="Directory containing Vox2 test audio, typically .../vox2_test_aac/aac",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        default="lists/train_list_vox2_aac12.txt",
        help="Stage 1 train list used to enforce speaker/file disjointness.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speaker-count", type=int, default=40)
    parser.add_argument("--target-trials", type=int, default=1000)
    parser.add_argument("--nontarget-trials", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="lists/vox2_test_monitor")
    return parser.parse_args()


def infer_speaker_key(rel_path: str) -> str:
    normalized = rel_path.replace("\\", "/")
    match = SPK_PATTERN.search(normalized)
    if not match:
        raise ValueError(f"Could not infer speaker key from path: {rel_path}")
    return match.group(0).lower()


def load_train_manifest(train_list_path: Path):
    train_speakers = set()
    train_paths = set()
    for raw in train_list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rel_path = line.split()[0].replace("\\", "/")
        train_paths.add(rel_path)
        train_speakers.add(infer_speaker_key(rel_path))
    return train_speakers, train_paths


def scan_monitor_audio(base_path: Path, monitor_dir: Path):
    entries = []
    for path in sorted(monitor_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _AUDIO_SUFFIXES:
            continue
        rel_path = path.relative_to(base_path).as_posix()
        speaker_key = infer_speaker_key(rel_path)
        entries.append(
            {
                "rel_path": rel_path,
                "speaker_key": speaker_key,
            }
        )
    if not entries:
        raise RuntimeError(f"No audio files found under monitor dir: {monitor_dir}")
    entries.sort(key=lambda item: (item["speaker_key"], item["rel_path"]))
    return entries


def group_by_speaker(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["speaker_key"]].append(entry["rel_path"])
    return {speaker: sorted(paths) for speaker, paths in sorted(grouped.items())}


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

    items = list(candidates)
    result = []
    while len(result) < count:
        cycle = list(items)
        rng.shuffle(cycle)
        take = min(count - len(result), len(cycle))
        result.extend(cycle[:take])
    return result


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    base_path = Path(args.base_path)
    monitor_dir = Path(args.monitor_dir)
    train_list_path = Path(args.train_list)
    output_dir = Path(args.output_dir)

    if not base_path.is_dir():
        raise FileNotFoundError(f"Base path not found: {base_path}")
    if not monitor_dir.is_dir():
        raise FileNotFoundError(f"Monitor dir not found: {monitor_dir}")
    if not train_list_path.is_file():
        raise FileNotFoundError(f"Train list not found: {train_list_path}")
    monitor_dir.relative_to(base_path)

    train_speakers, train_paths = load_train_manifest(train_list_path)
    entries = scan_monitor_audio(base_path, monitor_dir)
    by_speaker = group_by_speaker(entries)

    overlapping_speakers = sorted(set(by_speaker) & train_speakers)
    if overlapping_speakers:
        raise RuntimeError(
            "Monitor speakers overlap with Stage 1 train speakers. "
            f"Examples: {', '.join(overlapping_speakers[:5])}"
        )

    overlapping_paths = sorted(
        rel_path
        for rel_path in (entry["rel_path"] for entry in entries)
        if rel_path in train_paths
    )
    if overlapping_paths:
        raise RuntimeError(
            "Monitor audio paths overlap with Stage 1 train files. "
            f"Examples: {', '.join(overlapping_paths[:5])}"
        )

    eligible_speakers = sorted(
        speaker for speaker, paths in by_speaker.items() if len(paths) >= 3
    )
    if len(eligible_speakers) < args.speaker_count:
        raise RuntimeError(
            f"Not enough eligible monitor speakers ({len(eligible_speakers)}) "
            f"for requested speaker count {args.speaker_count}."
        )

    speakers = list(eligible_speakers)
    rng.shuffle(speakers)
    selected_speakers = sorted(speakers[: args.speaker_count])

    split_dir = output_dir
    split_dir.mkdir(parents=True, exist_ok=True)

    enroll_lines = []
    trials_lines = []
    manifest = {
        "purpose": "development-monitor-only",
        "seed": int(args.seed),
        "base_path": str(base_path.resolve()),
        "monitor_dir": str(monitor_dir.resolve()),
        "train_list": str(train_list_path.resolve()),
        "speaker_count": int(args.speaker_count),
        "selected_speakers": selected_speakers,
        "target_trials": int(args.target_trials),
        "nontarget_trials": int(args.nontarget_trials),
        "speakers": {},
    }

    positives_by_key = {}
    negatives_by_key = {}
    enroll_key_by_speaker = {}

    for speaker in selected_speakers:
        paths = list(by_speaker[speaker])
        enroll_path = paths[0]
        verify_paths = paths[1:]
        enroll_key = f"{speaker}-enroll"
        enroll_key_by_speaker[speaker] = enroll_key
        positives_by_key[enroll_key] = verify_paths
        negatives = []
        for other in selected_speakers:
            if other == speaker:
                continue
            negatives.extend(by_speaker[other][1:])
        negatives_by_key[enroll_key] = negatives
        enroll_lines.append(f"{enroll_key} {enroll_path}")
        manifest["speakers"][speaker] = {
            "enroll_key": enroll_key,
            "enroll_path": enroll_path,
            "num_audio": len(paths),
            "num_verify_audio": len(verify_paths),
        }

    enroll_keys = [enroll_key_by_speaker[speaker] for speaker in selected_speakers]
    target_quotas = allocate_quotas(enroll_keys, args.target_trials)
    nontarget_quotas = allocate_quotas(enroll_keys, args.nontarget_trials)

    for idx, enroll_key in enumerate(enroll_keys):
        for rel_path in sample_with_cycles(
            positives_by_key[enroll_key],
            target_quotas.get(enroll_key, 0),
            random.Random(args.seed + idx * 2 + 1),
        ):
            trials_lines.append(f"{enroll_key} {rel_path} 1")
        for rel_path in sample_with_cycles(
            negatives_by_key[enroll_key],
            nontarget_quotas.get(enroll_key, 0),
            random.Random(args.seed + idx * 2 + 2),
        ):
            trials_lines.append(f"{enroll_key} {rel_path} 0")

    random.Random(args.seed + 999).shuffle(trials_lines)

    (split_dir / "enroll.lst").write_text("\n".join(enroll_lines) + "\n", encoding="utf-8")
    (split_dir / "trials.lst").write_text("\n".join(trials_lines) + "\n", encoding="utf-8")
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Eligible monitor speakers: {len(eligible_speakers)}")
    print(f"Selected monitor speakers: {len(selected_speakers)}")
    print(f"Target trials: {args.target_trials}")
    print(f"Non-target trials: {args.nontarget_trials}")
    print(f"Output dir: {split_dir.resolve()}")


if __name__ == "__main__":
    main()
