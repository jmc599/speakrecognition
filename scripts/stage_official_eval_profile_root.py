import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


SPEAKER_ID_PATTERN = re.compile(r"(id\d+)", re.IGNORECASE)
DEFAULT_FLAT_ROOT = r"F:\speakerreg\rknn_eval_set_official_eval_inf\features"
DEFAULT_OUTPUT_ROOT = r"F:\speakerreg\artifacts\strict_profile_official_eval_inf\features"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage official-eval flat feature directories into a strict "
            "features/<speaker>/*.npy profile root."
        )
    )
    parser.add_argument(
        "--flat-root",
        type=str,
        default=DEFAULT_FLAT_ROOT,
        help="Flat official-eval feature root containing enroll/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Strict profile feature root to populate as features/<speaker>/*.npy.",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy"],
        help="Preferred staging mode. hardlink falls back to copy when linking fails.",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default="",
        help="Optional metadata JSON path. Defaults to <output-root>/../staging_metadata.json.",
    )
    return parser.parse_args()


def infer_speaker_id(path):
    candidates = [
        path.stem,
        path.name,
        str(path.relative_to(path.parents[1])) if len(path.parents) >= 2 else str(path),
        str(path),
    ]
    for text in candidates:
        match = SPEAKER_ID_PATTERN.search(text)
        if match:
            return match.group(1).lower()
    raise ValueError(f"Could not infer speaker id from feature path: {path}")


def default_metadata_output(output_root):
    return output_root.parent / "staging_metadata.json"


def ensure_clean_output_root(output_root):
    if output_root.exists():
        for path in sorted(output_root.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)


def staged_name_for(source_path, seen_names):
    name = source_path.name
    if name not in seen_names:
        seen_names.add(name)
        return name

    stem = source_path.stem
    suffix = source_path.suffix
    index = 2
    while True:
        candidate = f"{stem}__dup{index}{suffix}"
        if candidate not in seen_names:
            seen_names.add(candidate)
            return candidate
        index += 1


def materialize_file(source_path, target_path, preferred_mode):
    if target_path.exists():
        return "existing"

    if preferred_mode == "copy":
        shutil.copy2(source_path, target_path)
        return "copy"

    try:
        target_path.hardlink_to(source_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copy"


def collect_feature_files(flat_root):
    feature_files = [path for path in flat_root.rglob("*.npy") if path.is_file()]
    if not feature_files:
        raise RuntimeError(f"No .npy feature files found under: {flat_root}")
    return sorted(feature_files)


def main():
    args = parse_args()
    flat_root = Path(args.flat_root)
    output_root = Path(args.output_root)
    metadata_output = (
        Path(args.metadata_output)
        if args.metadata_output
        else default_metadata_output(output_root)
    )

    if not flat_root.is_dir():
        raise FileNotFoundError(f"Flat feature root not found: {flat_root}")

    feature_files = collect_feature_files(flat_root)
    ensure_clean_output_root(output_root)

    speaker_counts = defaultdict(int)
    mode_counts = defaultdict(int)
    seen_names_by_speaker = defaultdict(set)
    collisions = []

    for source_path in feature_files:
        speaker_id = infer_speaker_id(source_path)
        speaker_dir = output_root / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        staged_name = staged_name_for(source_path, seen_names_by_speaker[speaker_id])
        if staged_name != source_path.name:
            collisions.append(
                {
                    "speaker_id": speaker_id,
                    "source": str(source_path),
                    "staged_name": staged_name,
                }
            )

        target_path = speaker_dir / staged_name
        mode_used = materialize_file(source_path, target_path, args.link_mode)
        mode_counts[mode_used] += 1
        speaker_counts[speaker_id] += 1

    usable_speakers = {
        speaker_id: count for speaker_id, count in speaker_counts.items() if count >= 4
    }
    metadata = {
        "flat_root": str(flat_root),
        "output_root": str(output_root),
        "requested_link_mode": args.link_mode,
        "mode_counts": dict(sorted(mode_counts.items())),
        "num_input_files": len(feature_files),
        "num_staged_files": int(sum(mode_counts.values())),
        "num_speakers_total": len(speaker_counts),
        "num_speakers_usable_for_enroll_count_3": len(usable_speakers),
        "min_required_files_per_speaker": 4,
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "filename_collisions": collisions,
    }

    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Flat root: {flat_root}")
    print(f"Output root: {output_root}")
    print(f"Input .npy files: {len(feature_files)}")
    print(f"Staged files: {metadata['num_staged_files']}")
    print(f"Speakers: {metadata['num_speakers_total']}")
    print(
        "Usable speakers (>=4 files): "
        f"{metadata['num_speakers_usable_for_enroll_count_3']}"
    )
    print(f"Mode counts: {metadata['mode_counts']}")
    print(f"Metadata: {metadata_output}")


if __name__ == "__main__":
    main()
