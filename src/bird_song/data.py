from __future__ import annotations

from pathlib import Path


def resolve_dataset_root(project_root: Path, requested: Path | None = None) -> Path:
    root = (requested or project_root / "bird_songs_dataset").resolve()
    if not (root / "wavfiles").is_dir() or not (root / "bird_songs_metadata.csv").is_file():
        raise FileNotFoundError(
            f"Expected {root / 'wavfiles'} and {root / 'bird_songs_metadata.csv'}"
        )
    return root
