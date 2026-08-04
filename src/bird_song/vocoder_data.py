from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .vocoder import VocoderMelNormalizer, VocoderSpectrogramConfig


class VocoderSpectrogramDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Read cached raw BigVGAN log-mels and apply invertible training normalization."""

    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        split: str,
        classes: Iterable[str],
        config: VocoderSpectrogramConfig,
        normalizer: VocoderMelNormalizer,
    ) -> None:
        rows = pd.read_csv(manifest_path)
        required = {"split", "name", "relative_vocoder_mel_path"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Cache manifest is missing columns: {sorted(missing)}")
        self.rows = rows.loc[rows["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split {split!r}")
        self.cache_root = cache_root
        self.classes = tuple(classes)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        unknown = sorted(set(self.rows["name"]) - set(self.classes))
        if unknown:
            raise ValueError(f"Unknown classes in cache manifest: {unknown}")
        self.labels = self.rows["name"].map(self.class_to_index).astype(int).tolist()
        self.config = config
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[index]
        path = self.cache_root / Path(row["relative_vocoder_mel_path"])
        array = np.load(path, allow_pickle=False)
        expected = (self.config.n_mels, self.config.expected_frames)
        if array.shape != expected:
            raise ValueError(f"Expected raw log-mel {expected}, got {array.shape} in {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite raw log-mel values in {path}")
        raw = torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0)
        return self.normalizer.normalize(raw), self.labels[index], str(path)


def make_vocoder_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    *,
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
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )
