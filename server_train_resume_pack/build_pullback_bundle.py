from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable, List, Set, Tuple
import glob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tar.gz archive from pullback manifest files."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Manifest path relative to --root or absolute. May be passed multiple times.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root to resolve relative manifest entries against.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pullback_bundles",
        help="Directory to place the bundle and reports in.",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default="",
        help="Output bundle basename without extension. Defaults to the first manifest stem.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report entries without writing the tar.gz bundle.",
    )
    return parser.parse_args()


def has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[]")


def ensure_under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root)
    return resolved


def load_manifest_lines(manifest_path: Path, root: Path, seen_manifests: Set[Path]) -> List[str]:
    manifest_path = manifest_path.resolve()
    if manifest_path in seen_manifests:
        return []
    seen_manifests.add(manifest_path)

    lines: List[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("archive_manifests/") and manifest_path.parent == (root / "archive_manifests"):
            nested = root / line
            if nested.is_file():
                lines.extend(load_manifest_lines(nested, root, seen_manifests))
                continue
        lines.append(line)
    return lines


def resolve_pattern(entry: str, root: Path) -> Tuple[List[Path], List[str]]:
    entry = entry.replace("\\", "/")
    missing: List[str] = []
    matches: List[Path] = []

    if has_glob(entry):
        raw_matches = glob.glob(str(root / entry), recursive=True)
        if not raw_matches:
            missing.append(entry)
            return matches, missing
        for raw in raw_matches:
            path = Path(raw)
            try:
                matches.append(ensure_under_root(path, root))
            except ValueError:
                continue
        return matches, missing

    candidate = root / entry.rstrip("/")
    if not candidate.exists():
        missing.append(entry)
        return matches, missing

    try:
        matches.append(ensure_under_root(candidate, root))
    except ValueError:
        missing.append(entry)
    return matches, missing


def dedupe_paths(paths: Iterable[Path], root: Path) -> List[Path]:
    unique = sorted({p.resolve() for p in paths}, key=lambda p: (len(p.parts), str(p)))
    kept: List[Path] = []
    kept_dirs: List[Path] = []
    for path in unique:
        if any(parent == path or parent in path.parents for parent in kept_dirs):
            continue
        kept.append(path)
        if path.is_dir():
            kept_dirs.append(path)
    return kept


def write_report(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).strip()
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = []
    for raw_manifest in args.manifest:
        path = Path(raw_manifest)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        manifest_paths.append(path)

    manifest_entries: List[str] = []
    seen_manifests: Set[Path] = set()
    for manifest_path in manifest_paths:
        manifest_entries.extend(load_manifest_lines(manifest_path, root, seen_manifests))

    resolved_paths: List[Path] = []
    missing_entries: List[str] = []
    for entry in manifest_entries:
        matched, missing = resolve_pattern(entry, root)
        resolved_paths.extend(matched)
        missing_entries.extend(missing)

    selected_paths = dedupe_paths(resolved_paths, root)
    included_lines = [str(path.relative_to(root)).replace("\\", "/") for path in selected_paths]
    missing_lines = sorted(dict.fromkeys(missing_entries))

    bundle_name = args.bundle_name.strip() or manifest_paths[0].stem
    included_report = output_dir / f"{bundle_name}_included.txt"
    missing_report = output_dir / f"{bundle_name}_missing.txt"
    write_report(included_report, included_lines)
    write_report(missing_report, missing_lines)

    print(f"Root: {root}")
    print(f"Bundle name: {bundle_name}")
    print(f"Included paths: {len(included_lines)}")
    print(f"Missing entries: {len(missing_lines)}")
    print(f"Included report: {included_report}")
    print(f"Missing report: {missing_report}")

    if args.dry_run:
        print("Dry run only; bundle not written.")
        return

    bundle_path = output_dir / f"{bundle_name}.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in selected_paths:
            arcname = str(path.relative_to(root)).replace("\\", "/")
            tar.add(path, arcname=arcname, recursive=True)
    print(f"Bundle written: {bundle_path}")


if __name__ == "__main__":
    main()
