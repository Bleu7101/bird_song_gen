from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from bird_song.data import resolve_dataset_root
from bird_song.runtime import save_json
from bird_song.spectrogram_cache import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2}


def hash_audio(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def deduplicate_manifest_rows(
    rows: pd.DataFrame,
    dataset_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, str]], int]:
    required = {"split", "name", "id", "filename", "relative_wav_path"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if rows.empty or rows[list(required)].isna().any().any():
        raise ValueError("Manifest is empty or has missing required values")
    unknown_splits = sorted(set(rows["split"]) - set(SPLIT_PRIORITY))
    if unknown_splits:
        raise ValueError(f"Manifest has unsupported splits: {unknown_splits}")
    root = dataset_root.resolve()
    rows = rows.copy()
    audio_paths = []
    for value in rows["relative_wav_path"]:
        path = (root / Path(str(value))).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Audio path escapes dataset root: {value}")
        if not path.is_file():
            raise FileNotFoundError(f"Manifest audio file is missing: {path}")
        audio_paths.append(path)
    rows["audio_sha256"] = [hash_audio(path) for path in audio_paths]
    rows["_split_priority"] = rows["split"].map(SPLIT_PRIORITY)
    rows = rows.sort_values(
        ["audio_sha256", "_split_priority", "name", "id", "filename"],
        kind="stable",
    )
    duplicate_groups = [group for _, group in rows.groupby("audio_sha256", sort=True) if len(group) > 1]
    retained = rows.drop_duplicates("audio_sha256", keep="first").drop(columns=["_split_priority"])
    retained = retained.sort_values(["split", "name", "id", "filename"], kind="stable").reset_index(drop=True)
    if retained["audio_sha256"].duplicated().any():
        raise RuntimeError("Content-safe manifest still contains duplicate audio hashes")

    removed_records: list[dict[str, str]] = []
    retained_by_hash = retained.set_index("audio_sha256")
    for group in duplicate_groups:
        kept = retained_by_hash.loc[str(group.iloc[0]["audio_sha256"])]
        for row in group.itertuples(index=False):
            if str(row.relative_wav_path) == str(kept.relative_wav_path):
                continue
            removed_records.append(
                {
                    "audio_sha256": str(row.audio_sha256),
                    "removed_split": str(row.split),
                    "removed_relative_wav_path": str(row.relative_wav_path),
                    "retained_split": str(kept["split"]),
                    "retained_relative_wav_path": str(kept["relative_wav_path"]),
                }
            )
    return retained, removed_records, len(duplicate_groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create versioned manifests with one logical row per byte-identical audio clip."
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "manifests/content_safe_v2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    source_rows = pd.read_csv(args.manifest)
    retained, removed_records, duplicate_group_count = deduplicate_manifest_rows(source_rows, dataset_root)

    output_paths = {
        split: args.output_dir / f"full_dataset_{split}.csv" for split in ("train", "validation", "test")
    }
    output_paths["full"] = args.output_dir / "full_dataset_manifest.csv"
    output_paths["protocol"] = args.output_dir / "protocol.json"
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output files already exist under {args.output_dir}; pass --overwrite intentionally")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        retained[retained["split"] == split].to_csv(output_paths[split], index=False)
    retained.to_csv(output_paths["full"], index=False)

    original_counts = source_rows["split"].value_counts().sort_index()
    retained_counts = retained["split"].value_counts().sort_index()
    save_json(
        output_paths["protocol"],
        {
            "format_version": 2,
            "source_manifest": _portable(args.manifest),
            "source_manifest_sha256": sha256_file(args.manifest.resolve()),
            "deduplication_key": "SHA-256 of source WAV bytes",
            "retention_priority": ["test", "validation", "train"],
            "original_split_counts": {str(key): int(value) for key, value in original_counts.items()},
            "retained_split_counts": {str(key): int(value) for key, value in retained_counts.items()},
            "removed_row_count": int(len(source_rows) - len(retained)),
            "duplicate_group_count": int(duplicate_group_count),
            "removed_rows": removed_records,
            "note": "This version preserves the historical validation and test rows when exact audio also occurs in training.",
        },
    )
    print(f"Wrote {len(retained)} unique-content rows to {args.output_dir}")
    print(f"Removed {len(source_rows) - len(retained)} duplicate rows; split counts: {retained_counts.to_dict()}")


if __name__ == "__main__":
    main()
