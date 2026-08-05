from __future__ import annotations

from typing import Any

import torch


def _gradient_energy(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(f"Expected [batch, channels, height, width], got {tuple(images.shape)}")
    time = images[..., 1:] - images[..., :-1]
    frequency = images[:, :, 1:, :] - images[:, :, :-1, :]
    return time.square().mean(dim=(1, 2, 3)), frequency.square().mean(dim=(1, 2, 3))


def detail_metrics(real: torch.Tensor, generated: torch.Tensor) -> dict[str, float]:
    real_time, real_frequency = _gradient_energy(real.float())
    generated_time, generated_frequency = _gradient_energy(generated.float())
    real_time_mean = real_time.mean().clamp_min(1e-8)
    real_frequency_mean = real_frequency.mean().clamp_min(1e-8)
    real_std = real.float().flatten(1).std(dim=1).mean().clamp_min(1e-8)
    generated_std = generated.float().flatten(1).std(dim=1).mean()
    return {
        "real_time_gradient_energy": float(real_time.mean()),
        "real_frequency_gradient_energy": float(real_frequency.mean()),
        "generated_time_gradient_energy": float(generated_time.mean()),
        "generated_frequency_gradient_energy": float(generated_frequency.mean()),
        "time_detail_ratio": float(generated_time.mean() / real_time_mean),
        "frequency_detail_ratio": float(generated_frequency.mean() / real_frequency_mean),
        "sample_std_ratio": float(generated_std / real_std),
    }


def validate_generated(images: torch.Tensor, expected_shape: tuple[int, int, int] = (1, 80, 256)) -> dict[str, Any]:
    images = images.detach()
    if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
        raise ValueError(f"Expected generated images shaped (batch, {expected_shape}), got {tuple(images.shape)}")
    if not torch.isfinite(images).all():
        raise ValueError("Generated mels contain NaN or infinity")
    return {
        "shape": list(images.shape),
        "minimum": float(images.min()),
        "maximum": float(images.max()),
        "finite": True,
        "within_normalized_range": bool((images >= -1.001).all() and (images <= 1.001).all()),
        "saturated_fraction": float((images.abs() >= 0.999).float().mean()),
    }
