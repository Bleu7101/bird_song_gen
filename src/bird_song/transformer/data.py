from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CachedSpectrogramDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Read Stage 2 normalized log-mel arrays from the cache manifest."""

    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        split: str,
        classes: Iterable[str],
        image_size: int = 128,
        specaugment: bool = False,
    ) -> None:
        rows = pd.read_csv(manifest_path)
        required = {"split", "name", "relative_spectrogram_path"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Cache manifest is missing columns: {sorted(missing)}")
        self.rows = rows[rows["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Cache manifest has no rows for split {split!r}")
        self.cache_root = cache_root
        self.classes = tuple(classes)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        unknown = sorted(set(self.rows["name"]) - set(self.classes))
        if unknown:
            raise ValueError(f"Cache manifest has unknown classes: {unknown}")
        self.labels = self.rows["name"].map(self.class_to_index).astype(int).tolist()
        self.image_size = image_size
        self.specaugment = specaugment

    def __len__(self) -> int:
        return len(self.rows)

    def _mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = spectrogram.clone()
        if torch.rand(()) < 0.5:
            width = int(torch.randint(1, 13, ()).item())
            start = int(torch.randint(self.image_size - width + 1, ()).item())
            spectrogram[:, start : start + width, :] = -1.0
        if torch.rand(()) < 0.5:
            width = int(torch.randint(1, 17, ()).item())
            start = int(torch.randint(self.image_size - width + 1, ()).item())
            spectrogram[:, :, start : start + width] = -1.0
        return spectrogram

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[index]
        path = self.cache_root / Path(row["relative_spectrogram_path"])
        array = np.load(path, allow_pickle=False)
        spectrogram = torch.as_tensor(array, dtype=torch.float32).squeeze()
        if spectrogram.shape != (self.image_size, self.image_size):
            raise ValueError(f"Expected {(self.image_size, self.image_size)} in {path}, got {tuple(spectrogram.shape)}")
        if not torch.isfinite(spectrogram).all():
            raise ValueError(f"Spectrogram contains NaN or infinity: {path}")
        minimum, maximum = float(spectrogram.amin()), float(spectrogram.amax())
        if minimum < -1.05 or maximum > 1.05:
            raise ValueError(f"Expected normalized values in [-1, 1] in {path}, got [{minimum}, {maximum}]")
        spectrogram = spectrogram.clamp(-1.0, 1.0).unsqueeze(0)
        if self.specaugment:
            spectrogram = self._mask(spectrogram)
        return spectrogram, self.labels[index], str(path)


def make_cached_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    training: bool = False,
    seed: int | None = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers cannot be negative")
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training and len(dataset) >= batch_size,
        generator=generator,
    )
