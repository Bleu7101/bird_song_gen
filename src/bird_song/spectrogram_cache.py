from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SpectrogramConfig


CACHE_MANIFEST = "spectrogram_manifest.csv"
CACHE_METADATA = "cache_metadata.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().upper()


def cache_object_path(array_hash: str) -> Path:
    normalized = array_hash.lower()
    return Path("objects") / normalized[:2] / f"{normalized}.npy"


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

    path_hashes: dict[str, str] = {}
    referenced_paths: set[Path] = set()
    for relative in rows["relative_spectrogram_path"].astype(str).unique():
        path = resolve_cache_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Cached spectrogram is missing: {path}")
        referenced_paths.add(path)
        path_hashes[relative] = array_sha256(load_cache_array(path, config))

    audited = rows.copy()
    actual_hashes = audited["relative_spectrogram_path"].astype(str).map(path_hashes)
    if "spectrogram_sha256" in audited.columns:
        expected_hashes = audited["spectrogram_sha256"].astype(str).str.upper()
        mismatched = expected_hashes.ne(actual_hashes)
        if mismatched.any():
            row = audited.loc[mismatched].iloc[0]
            raise ValueError(
                "Cached spectrogram hash does not match its manifest entry: "
                f"{row['relative_spectrogram_path']}"
            )
    audited["spectrogram_sha256"] = actual_hashes
    hash_groups = audited.groupby("spectrogram_sha256", sort=True)
    duplicated = [group for _, group in hash_groups if len(group) > 1]
    cross_split = [group for group in duplicated if group["split"].astype(str).nunique() > 1]
    physical_paths = audited["relative_spectrogram_path"].astype(str).nunique()
    unique_objects = audited["spectrogram_sha256"].nunique()
    unreferenced_paths = {path.resolve() for path in root.rglob("*.npy")} - referenced_paths
    summary: dict[str, Any] = {
        "format_version": 1,
        "representation": "normalized_float32_128x128_minus1_plus1",
        "row_count": int(len(audited)),
        "physical_path_count": int(physical_paths),
        "unique_object_count": int(unique_objects),
        "duplicate_reference_count": int(len(audited) - unique_objects),
        "duplicate_content_group_count": int(len(duplicated)),
        "redundant_physical_file_count": int(physical_paths - unique_objects),
        "unreferenced_physical_file_count": int(len(unreferenced_paths)),
        "cross_split_duplicate_group_count": int(len(cross_split)),
        "cross_split_duplicate_row_count": int(sum(len(group) for group in cross_split)),
        "split_counts": {str(key): int(value) for key, value in audited["split"].value_counts().sort_index().items()},
        "class_counts": {str(key): int(value) for key, value in audited["name"].value_counts().sort_index().items()},
        "cross_split_duplicate_groups": [
            {
                "spectrogram_sha256": str(group.iloc[0]["spectrogram_sha256"]),
                "splits": sorted(group["split"].astype(str).unique().tolist()),
                "rows": group[["split", "name", "relative_wav_path"]].to_dict(orient="records"),
            }
            for group in cross_split
        ],
    }
    return audited, summary


def canonicalize_spectrogram_cache(
    cache_root: Path,
    config: SpectrogramConfig,
    config_path: Path,
    apply: bool = False,
) -> dict[str, Any]:
    root = cache_root.resolve()
    manifest_path = root / CACHE_MANIFEST
    manifest_hash_before = sha256_file(manifest_path)
    audited, before = audit_spectrogram_cache(root, config)
    original_audited = audited.copy()
    audited["relative_spectrogram_path"] = audited["spectrogram_sha256"].map(
        lambda value: cache_object_path(str(value)).as_posix()
    )
    plan = {
        **before,
        "cache_root": str(root),
        "planned_layout": "content_addressed_npy",
        "planned_object_count": int(audited["spectrogram_sha256"].nunique()),
        "apply": bool(apply),
    }
    if not apply:
        return plan

    original_rows = pd.read_csv(manifest_path)
    source_by_hash: dict[str, Path] = {}
    original_referenced_paths: set[Path] = set()
    for row in original_rows.itertuples(index=False):
        relative = str(row.relative_spectrogram_path)
        source = resolve_cache_path(root, relative)
        original_referenced_paths.add(source)
        digest = str(
            original_audited.loc[
                (original_audited["split"].astype(str) == str(row.split))
                & (original_audited["relative_wav_path"].astype(str) == str(row.relative_wav_path)),
                "spectrogram_sha256",
            ].iloc[0]
        )
        source_by_hash.setdefault(digest, source)

    target_paths: set[Path] = set()
    for digest, source in source_by_hash.items():
        relative_target = cache_object_path(digest)
        target = (root / relative_target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        target_digest = array_sha256(load_cache_array(target, config))
        if target_digest != digest:
            raise RuntimeError(f"Canonical object hash mismatch: {target}")
        target_paths.add(target)

    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    audited.to_csv(temporary_manifest, index=False)
    temporary_manifest.replace(manifest_path)

    # Only remove files referenced by the manifest we just canonicalized.
    # Unmanifested arrays may belong to another workflow and require an
    # explicit, separately reviewed prune operation.
    for path in original_referenced_paths - target_paths:
        if path.is_file():
            path.unlink()
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        if directory != root:
            try:
                directory.rmdir()
            except OSError:
                pass

    canonical_rows, after = audit_spectrogram_cache(root, config)
    metadata = {
        **after,
        "layout": "content_addressed_npy",
        "cache_root": str(root),
        "canonicalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256_before_canonicalization": manifest_hash_before,
        "manifest_sha256": sha256_file(manifest_path),
        "spectrogram_config": config.to_dict(),
        "spectrogram_config_sha256": sha256_file(config_path),
    }
    metadata_path = root / CACHE_METADATA
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    if len(canonical_rows) != len(original_rows):
        raise RuntimeError("Cache row count changed during canonicalization")
    return metadata
