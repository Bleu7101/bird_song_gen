from __future__ import annotations

from typing import Any

import numpy as np
import torch


def waveform_diagnostics(waveform: torch.Tensor, expected_samples: int) -> dict[str, Any]:
    values = waveform.detach().float().cpu().flatten()
    finite = bool(torch.isfinite(values).all())
    clipped = float((values.abs() >= 0.999).float().mean()) if values.numel() else 1.0
    rms = float(values.square().mean().sqrt()) if values.numel() else 0.0
    peak = float(values.abs().amax()) if values.numel() else 0.0
    return {
        "samples": int(values.numel()),
        "expected_samples": int(expected_samples),
        "length_valid": int(values.numel() == expected_samples),
        "finite": int(finite),
        "clipped_fraction": clipped,
        "rms": rms,
        "peak": peak,
        "silent": int(rms < 1e-4),
        "valid": int(finite and values.numel() == expected_samples and clipped <= 0.001),
    }


def _stft_magnitude(waveform: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, dtype=waveform.dtype, device=waveform.device)
    result = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    return result.abs().clamp_min(1e-7)


def multi_resolution_stft_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Return paired spectral-convergence and log-magnitude errors."""

    reference = reference.detach().float().cpu().flatten()
    candidate = candidate.detach().float().cpu().flatten()
    if reference.numel() != candidate.numel() or reference.numel() < 2_048:
        raise ValueError("reference and candidate must have equal, sufficiently long waveforms")
    spectral_convergence: list[float] = []
    log_magnitude_l1: list[float] = []
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        target = _stft_magnitude(reference, n_fft, hop)
        predicted = _stft_magnitude(candidate, n_fft, hop)
        spectral_convergence.append(float(torch.linalg.vector_norm(target - predicted) / torch.linalg.vector_norm(target)))
        log_magnitude_l1.append(float(torch.abs(torch.log(target) - torch.log(predicted)).mean()))
    return {
        "mrstft_spectral_convergence": float(np.mean(spectral_convergence)),
        "mrstft_log_magnitude_l1": float(np.mean(log_magnitude_l1)),
    }
