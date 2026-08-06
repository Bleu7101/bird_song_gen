from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .vocoder import VocoderMelScaler, VocoderSpectrogramConfig


class BigVGANMelDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        split: str,
        classes: Iterable[str],
        config: VocoderSpectrogramConfig,
        scaler: VocoderMelScaler,
        specaugment: bool = False,
    ) -> None:
        rows = pd.read_csv(manifest_path)
        required = {"split", "name", "relative_mel_path"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"cache manifest is missing columns: {sorted(missing)}")
        self.rows = rows.loc[rows["split"].astype(str).eq(split)].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"cache contains no rows for split {split!r}")
        self.cache_root = cache_root
        self.classes = tuple(classes)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        unknown = sorted(set(self.rows["name"]) - set(self.classes))
        if unknown:
            raise ValueError(f"cache contains unknown classes: {unknown}")
        self.labels = torch.tensor([self.class_to_index[name] for name in self.rows["name"]], dtype=torch.long)
        self.config = config
        self.scaler = scaler
        self.specaugment = specaugment

    def __len__(self) -> int:
        return len(self.rows)

    def _mask(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply light masks in the normalized mel domain during Transformer training."""

        mel = mel.clone()
        if torch.rand(()) < 0.5:
            width = int(torch.randint(1, min(13, self.config.n_mels + 1), ()).item())
            start = int(torch.randint(self.config.n_mels - width + 1, ()).item())
            mel[:, start : start + width, :] = -1.0
        if torch.rand(()) < 0.5:
            width = int(torch.randint(1, min(17, self.config.expected_frames + 1), ()).item())
            start = int(torch.randint(self.config.expected_frames - width + 1, ()).item())
            mel[:, :, start : start + width] = -1.0
        return mel

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows.iloc[index]
        path = self.cache_root / Path(str(row["relative_mel_path"]))
        array = np.load(path, allow_pickle=False)
        expected = (self.config.n_mels, self.config.expected_frames)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"invalid raw mel at {path}: shape={array.shape}")
        mel = self.scaler.normalize(torch.from_numpy(np.asarray(array, dtype=np.float32))).unsqueeze(0)
        if self.specaugment:
            mel = self._mask(mel)
        return mel, self.labels[index], str(path)


def make_mel_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    *,
    training: bool = False,
    seed: int | None = None,
) -> DataLoader:
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers cannot be negative")
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
        drop_last=training and len(dataset) >= batch_size,
    )
