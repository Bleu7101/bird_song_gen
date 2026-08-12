from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bird_song.config import DEFAULT_CLASSES


def _metrics(confusion: np.ndarray) -> tuple[float, float, list[float]]:
    true_positives = np.diag(confusion).astype(float)
    precision = np.divide(
        true_positives,
        confusion.sum(axis=0),
        out=np.zeros_like(true_positives),
        where=confusion.sum(axis=0) > 0,
    )
    recall = np.divide(
        true_positives,
        confusion.sum(axis=1),
        out=np.zeros_like(true_positives),
        where=confusion.sum(axis=1) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=precision + recall > 0,
    )
    accuracy = float(true_positives.sum() / max(int(confusion.sum()), 1))
    return accuracy, float(f1.mean()), recall.tolist()


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Evaluate a classifier for the maintained low-resource experiment."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((len(DEFAULT_CLASSES), len(DEFAULT_CLASSES)), dtype=np.int64)
    total_loss = 0.0
    total_items = 0
    for specs, labels, _ in loader:
        specs = specs.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(specs)
        predictions = logits.argmax(1).cpu().numpy()
        truth = labels.numpy()
        confusion += np.bincount(
            truth * len(DEFAULT_CLASSES) + predictions,
            minlength=len(DEFAULT_CLASSES) ** 2,
        ).reshape(confusion.shape)
        total_loss += float(criterion(logits, labels_device)) * labels.numel()
        total_items += labels.numel()
    accuracy, macro_f1, recall = _metrics(confusion)
    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_species_recall": dict(zip(DEFAULT_CLASSES, recall)),
        "confusion": confusion.tolist(),
        "sample_count": int(confusion.sum()),
    }


def select_ratio(means: dict[int, float], tie_tolerance: float = 0.002) -> int:
    """Select the smallest ratio within the validation tie band."""
    if not means:
        raise ValueError("At least one validation mean is required")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    best = max(float(value) for value in means.values())
    return min(int(ratio) for ratio, value in means.items() if best - float(value) <= tie_tolerance)
