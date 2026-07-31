from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .audio import LogMelTransform, load_generated_spectrogram, load_waveform
from .config import SpectrogramConfig


def resolve_dataset_root(project_root: Path, requested: Path | None = None) -> Path:
    root = (requested or project_root / "bird_songs_dataset").resolve()
    if not (root / "wavfiles").is_dir() or not (root / "bird_songs_metadata.csv").is_file():
        raise FileNotFoundError(
            f"Expected {root}/wavfiles and {root}/bird_songs_metadata.csv. "
            "The dataset should not have an extra nested bird_songs_dataset folder."
        )
    return root


class ManifestDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        classes: Iterable[str],
        config: SpectrogramConfig,
        training: bool = False,
    ) -> None:
        self.rows = pd.read_csv(manifest_path)
        required = {"name", "relative_wav_path"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing)}")
        if self.rows.empty:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        self.dataset_root = dataset_root
        self.classes = tuple(classes)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        unknown = sorted(set(self.rows["name"]) - set(self.classes))
        if unknown:
            raise ValueError(f"Manifest has unknown classes: {unknown}")
        self.labels = self.rows["name"].map(self.class_to_index).astype(int).tolist()
        self.config = config
        self.training = training
        self.transform = LogMelTransform(config, training=training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[index]
        path = self.dataset_root / Path(row["relative_wav_path"])
        waveform = load_waveform(path, self.config, self.training)
        return self.transform(waveform), self.labels[index], str(path)

    def sample_weights(self) -> torch.Tensor:
        counts = torch.bincount(torch.tensor(self.labels), minlength=len(self.classes)).float()
        return (1.0 / counts.clamp_min(1.0))[torch.tensor(self.labels)]


class InferenceDataset(Dataset[tuple[torch.Tensor, str]]):
    AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}

    def __init__(self, paths: list[Path], config: SpectrogramConfig) -> None:
        self.paths = paths
        self.config = config
        self.transform = LogMelTransform(config, training=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        if path.suffix.lower() == ".npy":
            spec = load_generated_spectrogram(path, self.config)
        elif path.suffix.lower() in self.AUDIO_EXTENSIONS:
            spec = self.transform(load_waveform(path, self.config, training=False))
        else:
            raise ValueError(f"Unsupported input type: {path}")
        return spec, str(path)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    training: bool = False,
    balanced: bool = False,
    seed: int | None = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if workers < 0:
        raise ValueError("workers cannot be negative")
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    sampler = None
    shuffle = training
    if balanced:
        if not isinstance(dataset, ManifestDataset):
            raise TypeError("Balanced sampling requires ManifestDataset")
        sampler = WeightedRandomSampler(
            dataset.sample_weights(),
            len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training and len(dataset) >= batch_size,
    )
