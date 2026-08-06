from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from bird_song.config import DEFAULT_CLASSES
from bird_song.data import resolve_dataset_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create recording-ID-isolated train/validation/test manifests.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--search-seeds", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Print the selected split without writing files.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def group_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_indices, temporary_indices = next(first.split(frame, groups=frame["id"]))
    train = frame.iloc[train_indices].copy()
    temporary = frame.iloc[temporary_indices].copy()
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    validation_indices, test_indices = next(second.split(temporary, groups=temporary["id"]))
    return train, temporary.iloc[validation_indices].copy(), temporary.iloc[test_indices].copy()


def balance_score(full: pd.DataFrame, splits: tuple[pd.DataFrame, ...]) -> float:
    expected = full["name"].value_counts(normalize=True).reindex(DEFAULT_CLASSES).fillna(0)
    return sum(
        float((split["name"].value_counts(normalize=True).reindex(DEFAULT_CLASSES).fillna(0) - expected).abs().sum())
        for split in splits
    )


def prepare_manifest(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    output = frame.copy()
    output["split"] = split_name
    output["species_slug"] = output["name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    output["relative_wav_path"] = output["filename"].map(lambda name: (Path("wavfiles") / name).as_posix())
    columns = [
        "split", "name", "species_slug", "genus", "species", "id", "filename",
        "relative_wav_path", "source_url", "license", "recordist", "date", "sound_type",
    ]
    return output[[column for column in columns if column in output]].sort_values(["name", "id", "filename"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.search_seeds < 1:
        raise ValueError("--search-seeds must be at least 1")
    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    metadata = pd.read_csv(dataset_root / "bird_songs_metadata.csv")
    selected = metadata[metadata["name"].isin(DEFAULT_CLASSES)].copy().reset_index(drop=True)
    missing_classes = sorted(set(DEFAULT_CLASSES) - set(selected["name"]))
    if missing_classes:
        raise ValueError(f"Dataset is missing target species: {missing_classes}")

    candidates = []
    for seed in range(args.search_seeds):
        splits = group_split(selected, seed)
        candidates.append((balance_score(selected, splits), seed, splits))
    score, seed, (train, validation, test) = min(candidates, key=lambda item: (item[0], item[1]))
    id_sets = [set(frame["id"].astype(str)) for frame in (train, validation, test)]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise RuntimeError("Recording-ID leakage detected")

    manifests = {
        "train": prepare_manifest(train, "train"),
        "validation": prepare_manifest(validation, "validation"),
        "test": prepare_manifest(test, "test"),
    }
    summary = pd.concat(
        [frame.groupby("name").size().rename(split_name) for split_name, frame in manifests.items()], axis=1
    ).fillna(0).astype(int)
    print(f"Selected split seed: {seed}; balance score: {score:.6f}")
    print(summary.to_string())
    if args.dry_run:
        print("Dry run complete; no manifests were written.")
        return

    paths = {name: args.output_dir / f"full_dataset_{name}.csv" for name in manifests}
    paths["full"] = args.output_dir / "full_dataset_manifest.csv"
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Manifest files already exist. Pass --overwrite to replace them intentionally.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in manifests.items():
        frame.to_csv(paths[name], index=False)
    pd.concat(manifests.values(), ignore_index=True).to_csv(paths["full"], index=False)
    print(f"Wrote manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
