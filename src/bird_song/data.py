from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .audio import LogMelTransform, load_generated_spectrogram, load_waveform
from .config import SpectrogramConfig
from .spectrogram_cache import load_cache_array, resolve_cache_path


def resolve_dataset_root(project_root: Path, requested: Path | None = None) -> Path:
    root = (requested or project_root / "bird_songs_dataset").resolve()
    if not (root / "wavfiles").is_dir() or not (root / "bird_songs_metadata.csv").is_file():
        raise FileNotFoundError(
            f"Expected {root}/wavfiles and {root}/bird_songs_metadata.csv. "
            "The dataset should not have an extra nested bird_songs_dataset folder."
        )
    return root


def resolve_spectrogram_cache_root(project_root: Path, requested: Path | None = None) -> Path:
    root = (requested or project_root / "artifacts/spectrograms").resolve()
    manifest = root / "spectrogram_manifest.csv"
    if not root.is_dir() or not manifest.is_file():
        raise FileNotFoundError(
            f"Expected cached spectrograms and {manifest}. "
            "Generate or provide the historical real-audio spectrogram cache."
        )
    return root


class ManifestDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path | None,
        classes: Iterable[str],
        config: SpectrogramConfig,
        training: bool = False,
        spectrogram_cache_root: Path | None = None,
    ) -> None:
        self.rows = pd.read_csv(manifest_path)
        required = {"name", "relative_wav_path"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing)}")
        if self.rows.empty:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        self.classes = tuple(classes)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        unknown = sorted(set(self.rows["name"]) - set(self.classes))
        if unknown:
            raise ValueError(f"Manifest has unknown classes: {unknown}")
        self.labels = self.rows["name"].map(self.class_to_index).astype(int).tolist()
        self.config = config
        self.training = training
        self.dataset_root = dataset_root.resolve() if dataset_root is not None else None
        self.spectrogram_paths: list[Path] | None = None
        if spectrogram_cache_root is not None:
            cache_root = spectrogram_cache_root.resolve()
            cache_manifest_path = cache_root / "spectrogram_manifest.csv"
            cache_rows = pd.read_csv(cache_manifest_path)
            cache_required = {"split", "relative_wav_path", "relative_spectrogram_path"}
            cache_missing = cache_required - set(cache_rows.columns)
            if cache_missing:
                raise ValueError(f"Cached manifest {cache_manifest_path} is missing columns: {sorted(cache_missing)}")
            if cache_rows[list(cache_required)].isna().any().any():
                raise ValueError(f"Cached manifest {cache_manifest_path} has missing required values")
            if "split" not in self.rows.columns:
                raise ValueError(f"Manifest {manifest_path} needs a split column for cached spectrogram lookup")

            def key(frame: pd.DataFrame) -> pd.Series:
                return frame["split"].astype(str) + "\0" + frame["relative_wav_path"].astype(str)

            cache_keys = key(cache_rows)
            if cache_keys.duplicated().any():
                raise ValueError(f"Cached manifest has duplicate recording paths: {cache_manifest_path}")
            cache_index = dict(zip(cache_keys.tolist(), cache_rows["relative_spectrogram_path"].tolist()))
            row_keys = key(self.rows)
            missing_keys = [value for value in row_keys.tolist() if value not in cache_index]
            if missing_keys:
                raise FileNotFoundError(
                    f"Cached manifest {cache_manifest_path} is missing {len(missing_keys)} rows from {manifest_path}"
                )
            self.spectrogram_paths = [
                resolve_cache_path(cache_root, str(cache_index[value])) for value in row_keys.tolist()
            ]
            missing_paths = [path for path in self.spectrogram_paths if not path.is_file()]
            if missing_paths:
                raise FileNotFoundError(f"Cached spectrogram file is missing: {missing_paths[0]}")
            self.transform = None
        else:
            if self.dataset_root is None:
                raise ValueError("dataset_root is required when spectrogram_cache_root is not provided")
            self.transform = LogMelTransform(config, training=training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[index]
        if self.spectrogram_paths is not None:
            path = self.spectrogram_paths[index]
            spec = torch.from_numpy(load_cache_array(path, self.config))
            return spec.unsqueeze(0), self.labels[index], str(path)

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
