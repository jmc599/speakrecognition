import argparse
import re
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
SPK_PATTERN = re.compile(r"id\d+", flags=re.IGNORECASE)


def infer_speaker_key(file_path: Path, dataset_root: Path) -> str:
    rel_parts = file_path.relative_to(dataset_root).parts
    for part in rel_parts:
        if re.fullmatch(r"id\d+", part, flags=re.IGNORECASE):
            return part.lower()

    stem = file_path.stem
    m = re.match(r"(id\d+)[-_].*", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return file_path.parent.name.lower()


def extract_speakers_from_text_file(path: Path) -> set[str]:
    speakers = set()
    if not path.is_file():
        return speakers

    for line in path.read_text(encoding="utf-8").splitlines():
        for token in line.strip().split():
            m = SPK_PATTERN.search(token.replace("\\", "/"))
            if m:
                speakers.add(m.group(0).lower())
    return speakers


def build_open_set_list(
    dataset_root: Path,
    include_subdir: str,
    eval_trials: Path,
    eval_enroll: Path,
    output_list: Path,
):
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    target_root = dataset_root / include_subdir if include_subdir else dataset_root
    if not target_root.exists():
        raise FileNotFoundError(f"target data folder not found: {target_root}")

    eval_speakers = set()
    eval_speakers.update(extract_speakers_from_text_file(eval_trials))
    eval_speakers.update(extract_speakers_from_text_file(eval_enroll))
    if not eval_speakers:
        raise RuntimeError(
            "No eval speakers parsed from eval lists. "
            "Check --eval-trials and --eval-enroll."
        )

    wavs = []
    for p in target_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            wavs.append(p)
    wavs.sort()
    if not wavs:
        raise RuntimeError(f"no audio files found under: {target_root}")

    kept = []
    skipped = 0
    for wav in wavs:
        spk_key = infer_speaker_key(wav, dataset_root)
        if spk_key in eval_speakers:
            skipped += 1
            continue
        kept.append((wav, spk_key))

    if not kept:
        raise RuntimeError("All files were filtered out. Check dataset paths and eval lists.")

    spk_to_id: dict[str, int] = {}
    lines = []
    for wav, spk_key in kept:
        if spk_key not in spk_to_id:
            spk_to_id[spk_key] = len(spk_to_id)
        rel_path = wav.relative_to(dataset_root).as_posix()
        lines.append(f"{rel_path} {spk_to_id[spk_key]}")

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "num_eval_speakers": len(eval_speakers),
        "num_total_files": len(wavs),
        "num_kept_files": len(kept),
        "num_skipped_files": skipped,
        "num_kept_speakers": len(spk_to_id),
        "output_list": str(output_list.resolve()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train list from data/ while excluding speakers present in eval lists."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--include-subdir",
        type=str,
        default="data",
        help="Subfolder under dataset root to scan for training audio.",
    )
    parser.add_argument(
        "--eval-trials",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval\lists\trials.lst",
        help="Eval trials list path.",
    )
    parser.add_argument(
        "--eval-enroll",
        type=str,
        default=r"F:\cn_celeb\cn-celeb_v2\CN-Celeb_flac\eval\lists\enroll.lst",
        help="Eval enroll list path.",
    )
    parser.add_argument(
        "--output-list",
        type=str,
        default="lists/train_list_open_set.txt",
        help="Output train list path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = build_open_set_list(
        dataset_root=Path(args.dataset_root),
        include_subdir=args.include_subdir.strip(),
        eval_trials=Path(args.eval_trials),
        eval_enroll=Path(args.eval_enroll),
        output_list=Path(args.output_list),
    )
    print(f"Eval speakers excluded: {stats['num_eval_speakers']}")
    print(f"Total files scanned: {stats['num_total_files']}")
    print(f"Kept files: {stats['num_kept_files']}")
    print(f"Skipped files: {stats['num_skipped_files']}")
    print(f"Kept speakers: {stats['num_kept_speakers']}")
    print(f"Output list: {stats['output_list']}")


if __name__ == "__main__":
    main()
