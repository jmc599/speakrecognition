import argparse
import os
import re
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
SPK_PATTERN = re.compile(r"^id\d+$", re.IGNORECASE)


def _infer_speaker_key(file_path: Path, dataset_root: Path) -> str:
    rel_parts = file_path.relative_to(dataset_root).parts
    for part in rel_parts:
        if SPK_PATTERN.match(part):
            return part.lower()

    stem = file_path.stem
    m = re.match(r"(id\d+)[-_].*", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return file_path.parent.name.lower()


def build_speaker_list(
    dataset_root: str,
    output_list: str,
    include_subdir: str = "data",
) -> tuple[int, int]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    target_root = root / include_subdir if include_subdir else root
    if not target_root.exists():
        raise FileNotFoundError(f"target data folder not found: {target_root}")

    wavs = []
    for p in target_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            wavs.append(p)
    wavs.sort()

    if not wavs:
        raise RuntimeError(f"no audio files found under: {target_root}")

    spk_to_id: dict[str, int] = {}
    lines = []
    for wav in wavs:
        spk_key = _infer_speaker_key(wav, root)
        if spk_key not in spk_to_id:
            spk_to_id[spk_key] = len(spk_to_id)
        rel_path = wav.relative_to(root).as_posix()
        lines.append(f"{rel_path} {spk_to_id[spk_key]}")

    output_path = Path(output_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines), len(spk_to_id)


def main():
    parser = argparse.ArgumentParser(
        description="Build 'relative_path speaker_id' list for speaker training."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Dataset root path. Example: F:/cn_celeb/CN-Celeb_flac",
    )
    parser.add_argument(
        "--output-list",
        type=str,
        default="lists/train_list.txt",
        help="Output list path.",
    )
    parser.add_argument(
        "--include-subdir",
        type=str,
        default="data",
        help="Only scan this subfolder under dataset root. Set empty string to scan all.",
    )
    args = parser.parse_args()

    include_subdir = args.include_subdir.strip()
    total, num_spk = build_speaker_list(
        dataset_root=args.dataset_root,
        output_list=args.output_list,
        include_subdir=include_subdir,
    )
    print(f"Saved list: {os.path.abspath(args.output_list)}")
    print(f"Total audio files: {total}")
    print(f"Total speakers: {num_spk}")


if __name__ == "__main__":
    main()
