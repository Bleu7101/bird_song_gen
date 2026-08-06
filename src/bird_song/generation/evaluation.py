from __future__ import annotations

from typing import Any

import torch


def _gradient_energy(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(f"Expected images shaped (batch, channels, height, width), got {tuple(images.shape)}")
    time = images[..., 1:] - images[..., :-1]
    frequency = images[:, :, 1:, :] - images[:, :, :-1, :]
    return time.square().mean(dim=(1, 2, 3)), frequency.square().mean(dim=(1, 2, 3))


def detail_metrics(real: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    """Compare local time/frequency detail without requiring the classifier."""
    real_time, real_frequency = _gradient_energy(real.float())
    generated_time, generated_frequency = _gradient_energy(generated.float())
    real_time_mean = real_time.mean().clamp_min(1e-8)
    real_frequency_mean = real_frequency.mean().clamp_min(1e-8)
    return {
        "real_time_gradient_energy": float(real_time.mean()),
        "real_frequency_gradient_energy": float(real_frequency.mean()),
        "generated_time_gradient_energy": float(generated_time.mean()),
        "generated_frequency_gradient_energy": float(generated_frequency.mean()),
        "time_detail_ratio": float(generated_time.mean() / real_time_mean),
        "frequency_detail_ratio": float(generated_frequency.mean() / real_frequency_mean),
    }


def diversity_metrics(images: torch.Tensor) -> dict[str, float]:
    """Estimate sample diversity and flag exact/near-exact collapse."""
    if images.ndim != 4 or images.shape[0] < 2:
        raise ValueError("At least two images are required for diversity metrics")
    flattened = images.float().flatten(1)
    distances = torch.pdist(flattened)
    return {
        "pairwise_l2_mean": float(distances.mean()),
        "pairwise_l2_median": float(distances.median()),
        "pairwise_l2_min": float(distances.min()),
        "sample_std": float(flattened.std()),
        "finite": float(torch.isfinite(images).all()),
        "within_range": float(((images >= -1.001) & (images <= 1.001)).all()),
    }


def validate_generated(images: torch.Tensor, channels: int = 1, image_size: int = 128) -> dict[str, Any]:
    images = images.detach()
    expected = (channels, image_size, image_size)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected:
        raise ValueError(f"Expected generated images shaped (batch, {expected}), got {tuple(images.shape)}")
    if not torch.isfinite(images).all():
        raise ValueError("Generated spectrograms contain NaN or infinity")
    return {
        "shape": list(images.shape),
        "minimum": float(images.min()),
        "maximum": float(images.max()),
        "finite": True,
        "within_normalized_range": bool((images >= -1.001).all() and (images <= 1.001).all()),
    }
