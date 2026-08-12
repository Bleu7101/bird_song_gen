from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SpectrogramConfig


CACHE_MANIFEST = "spectrogram_manifest.csv"


def cache_array_path(row_index: int, filename: str) -> Path:
    """Return a stable, readable cache path for one manifest row."""
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(filename).stem).strip("._") or "clip"
    return Path("arrays") / f"{row_index:05d}_{stem}.npy"


def load_cache_array(path: Path, config: SpectrogramConfig) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)
    expected = (config.n_mels, config.spectrogram_width)
    if array.shape != expected:
        raise ValueError(f"Cached spectrogram must be {expected}, got {array.shape} at {path}")
    if array.dtype != np.float32:
        raise ValueError(f"Cached spectrogram must be float32, got {array.dtype} at {path}")
    if not np.isfinite(array).all() or float(array.min()) < -1.00001 or float(array.max()) > 1.00001:
        raise ValueError(f"Cached spectrogram is non-finite or outside [-1,1]: {path}")
    return array


def resolve_cache_path(cache_root: Path, relative_path: str) -> Path:
    root = cache_root.resolve()
    path = (root / Path(relative_path)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Cache path escapes {root}: {relative_path}")
    return path


def audit_spectrogram_cache(
    cache_root: Path,
    config: SpectrogramConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a cache manifest and every referenced array."""
    root = cache_root.resolve()
    manifest_path = root / CACHE_MANIFEST
    rows = pd.read_csv(manifest_path)
    required = {"split", "name", "relative_wav_path", "relative_spectrogram_path"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Cache manifest is missing columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError(f"Cache manifest is empty: {manifest_path}")
    if rows[list(required)].isna().any().any():
        raise ValueError(f"Cache manifest has missing required values: {manifest_path}")
    for column in required:
        if rows[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Cache manifest has blank {column} values: {manifest_path}")
    logical_keys = rows["split"].astype(str) + "\0" + rows["relative_wav_path"].astype(str)
    if logical_keys.duplicated().any():
        raise ValueError(f"Cache manifest has duplicate split/path keys: {manifest_path}")

    referenced_paths: set[Path] = set()
    for relative in rows["relative_spectrogram_path"].astype(str).unique():
        path = resolve_cache_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Cached spectrogram is missing: {path}")
        load_cache_array(path, config)
        referenced_paths.add(path)

    unreferenced_paths = {path.resolve() for path in root.rglob("*.npy")} - referenced_paths
    summary: dict[str, Any] = {
        "format_version": 2,
        "representation": "normalized_float32_128x128_minus1_plus1",
        "row_count": int(len(rows)),
        "physical_path_count": int(rows["relative_spectrogram_path"].astype(str).nunique()),
        "unreferenced_physical_file_count": int(len(unreferenced_paths)),
        "split_counts": {str(key): int(value) for key, value in rows["split"].value_counts().sort_index().items()},
        "class_counts": {str(key): int(value) for key, value in rows["name"].value_counts().sort_index().items()},
    }
    return rows.copy(), summary
