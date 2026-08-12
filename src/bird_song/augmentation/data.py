from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bird_song.config import SpectrogramConfig
from bird_song.spectrogram_cache import load_cache_array, resolve_cache_path


def _normalize_relative_path(value: object) -> str:
    return str(value).replace("\\", "/")


def _validated_manifest(
    manifest_path: Path,
    cache_root: Path,
    classes: Iterable[str],
) -> tuple[pd.DataFrame, tuple[str, ...], list[Path]]:
    class_names = tuple(classes)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("classes must be non-empty and unique")
    rows = pd.read_csv(manifest_path)
    required = {"species", "relative_path", "pool_rank"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Generated manifest is missing columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError(f"Generated manifest is empty: {manifest_path}")
    if rows[list(required)].isna().any().any():
        raise ValueError(f"Generated manifest has missing required values: {manifest_path}")
    rows = rows.copy()
    rows["species"] = rows["species"].astype(str).str.strip()
    rows["relative_path"] = rows["relative_path"].map(_normalize_relative_path)
    if rows["species"].eq("").any() or rows["relative_path"].str.strip().eq("").any():
        raise ValueError(f"Generated manifest has blank required values: {manifest_path}")
    try:
        numeric_ranks = pd.to_numeric(rows["pool_rank"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Generated manifest has non-numeric pool ranks: {manifest_path}") from error
    rank_values = numeric_ranks.to_numpy(dtype=float)
    if not np.isfinite(rank_values).all() or not np.equal(rank_values, np.floor(rank_values)).all():
        raise ValueError(f"Generated manifest pool ranks must be finite integers: {manifest_path}")
    rows["pool_rank"] = numeric_ranks.astype(int)
    if rows["pool_rank"].lt(0).any():
        raise ValueError(f"Generated manifest pool ranks must be non-negative: {manifest_path}")
    if rows["relative_path"].duplicated().any():
        raise ValueError(f"Generated manifest has duplicate paths: {manifest_path}")
    if rows.duplicated(["species", "pool_rank"]).any():
        raise ValueError(f"Generated manifest has duplicate species/pool ranks: {manifest_path}")
    unknown = sorted(set(rows["species"]) - set(class_names))
    if unknown:
        raise ValueError(f"Generated manifest has unknown species: {unknown}")
    missing_species = sorted(set(class_names) - set(rows["species"]))
    if missing_species:
        raise ValueError(f"Generated manifest is missing species: {missing_species}")
    for species, group in rows.groupby("species", sort=False):
        ranks = sorted(group["pool_rank"].astype(int).tolist())
        if ranks != list(range(len(ranks))):
            raise ValueError(f"Generated manifest ranks for {species} must be contiguous from zero")

    root = cache_root.resolve()
    paths = [resolve_cache_path(root, value) for value in rows["relative_path"]]
    if len(set(paths)) != len(paths):
        raise ValueError(f"Generated manifest resolves multiple rows to the same path: {manifest_path}")
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Generated spectrogram is missing: {missing_paths[0]}")
    return rows, class_names, paths


def audit_generated_pool(
    manifest_path: Path,
    cache_root: Path,
    classes: Iterable[str],
    config: SpectrogramConfig,
) -> dict[str, object]:
    """Verify every generated array once before a sweep."""
    rows, class_names, paths = _validated_manifest(manifest_path, cache_root, classes)
    for path in paths:
        load_cache_array(path, config)
    counts = rows["species"].value_counts().reindex(class_names, fill_value=0)
    if counts.nunique() != 1:
        raise ValueError(f"Generated pool is not balanced by species: {counts.to_dict()}")
    return {
        "manifest": str(manifest_path.resolve()),
        "rows": int(len(rows)),
        "validated_arrays": int(len(paths)),
        "rows_per_species": {str(name): int(counts[name]) for name in class_names},
        "max_ratio_per_species": int(counts.iloc[0]),
    }


class GeneratedSpectrogramDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Strict classifier-ready generated arrays selected by per-species pool rank."""

    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        classes: Iterable[str],
        config: SpectrogramConfig,
        ratio_per_species: int,
        training: bool = False,
    ) -> None:
        if ratio_per_species < 1:
            raise ValueError("ratio_per_species must be at least 1")
        rows, self.classes, all_paths = _validated_manifest(manifest_path, cache_root, classes)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        selected = rows[rows["pool_rank"] < ratio_per_species].copy()
        counts = selected["species"].value_counts().reindex(self.classes, fill_value=0)
        incomplete = {name: int(count) for name, count in counts.items() if int(count) != ratio_per_species}
        if incomplete:
            raise ValueError(
                f"Expected {ratio_per_species} generated rows per species in {manifest_path}; got {incomplete}"
            )
        self.rows = selected.sort_values(["species", "pool_rank"], kind="stable").reset_index(drop=True)
        self.cache_root = cache_root.resolve()
        self.config = config
        self.training = training
        self.labels = [self.class_to_index[name] for name in self.rows["species"]]
        path_by_relative = dict(zip(rows["relative_path"].tolist(), all_paths, strict=True))
        self.paths = [path_by_relative[value] for value in self.rows["relative_path"]]

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _mask(spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()
        if torch.rand(()) < 0.5:
            height = int(spec.shape[-2])
            width = int(torch.randint(0, min(12, height) + 1, ()).item())
            if width:
                start = int(torch.randint(0, height - width + 1, ()).item())
                spec[:, start : start + width, :] = -1.0
        if torch.rand(()) < 0.5:
            length = int(spec.shape[-1])
            width = int(torch.randint(0, min(16, length) + 1, ()).item())
            if width:
                start = int(torch.randint(0, length - width + 1, ()).item())
                spec[:, :, start : start + width] = -1.0
        return spec

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path = self.paths[index]
        spec = torch.from_numpy(load_cache_array(path, self.config)).unsqueeze(0)
        if self.training:
            spec = self._mask(spec)
        return spec, self.labels[index], str(path)
