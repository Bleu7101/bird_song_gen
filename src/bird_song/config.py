from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpectrogramConfig:
    """Audio representation shared by real and generated-song classifiers."""

    sample_rate: int = 22_050
    duration_seconds: float = 3.0
    n_fft: int = 1_024
    hop_length: int = 512
    n_mels: int = 128
    spectrogram_width: int = 128
    f_min: float = 150.0
    f_max: float = 10_500.0
    top_db: float = 80.0

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate * self.duration_seconds))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SpectrogramConfig":
        return cls(**values)

    @classmethod
    def from_json(cls, path: Path) -> "SpectrogramConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


DEFAULT_CLASSES = ("American Robin", "Northern Cardinal", "Song Sparrow")
