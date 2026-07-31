from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import SpectrogramConfig
from .classifier.model import build_classifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, tuple[str, ...], SpectrogramConfig, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = tuple(checkpoint["classes"])
    config = SpectrogramConfig.from_dict(checkpoint["spectrogram_config"])
    model_config = dict(checkpoint["model_config"])
    # Version-1 checkpoints predate architecture selection and are residual CNNs.
    architecture = model_config.pop("architecture", checkpoint.get("architecture", "residual_cnn"))
    model = build_classifier(architecture=architecture, **model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, classes, config, checkpoint


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    """Write a checkpoint atomically so an interruption cannot corrupt the last good file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
