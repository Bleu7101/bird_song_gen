from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def make_cached_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    *,
    training: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Build a deterministic loader for the cached BigVGAN mel arrays."""

    if batch_size < 1 or workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative")
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
