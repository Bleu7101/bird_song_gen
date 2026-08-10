from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ..config import SpectrogramConfig


def normalized_logmel_to_waveform(
    spectrogram: np.ndarray,
    config: SpectrogramConfig,
    iterations: int = 32,
    target_peak: float = 0.95,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Invert normalized log-mel arrays with the fixed Griffin-Lim baseline.

    ``device`` controls where the mel inversion and Griffin-Lim iterations run.
    Keeping the default on CPU preserves the original API while allowing the
    reconstruction diagnostic to use CUDA when available.
    """
    array = np.asarray(spectrogram, dtype=np.float32).squeeze()
    expected = (config.n_mels, config.spectrogram_width)
    if array.shape != expected:
        raise ValueError(f"Expected normalized log-mel shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Spectrogram contains NaN or infinity")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target_peak must be in (0, 1]")
    decode_device = torch.device(device)
    # normalize_logmel maps relative dB in [-top_db, 0] to [-1, 1].
    relative_db = (np.clip(array, -1.0, 1.0) - 1.0) * (config.top_db / 2.0)
    mel_power = torch.from_numpy(np.power(10.0, relative_db / 10.0)).float().to(decode_device)
    mel_filter = torchaudio.functional.melscale_fbanks(
        n_freqs=config.n_fft // 2 + 1,
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels,
        sample_rate=config.sample_rate,
        norm=None,
        mel_scale="htk",
    ).to(decode_device)
    # Forward mel power is filter.T @ linear power. The pseudoinverse gives a
    # stable nonnegative least-squares-like approximation for Griffin-Lim.
    linear_power = torch.linalg.pinv(mel_filter.t()) @ mel_power
    linear_power = linear_power.clamp_min(0.0)
    waveform = torchaudio.functional.griffinlim(
        linear_power,
        window=torch.hann_window(config.n_fft, device=decode_device),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.n_fft,
        power=2.0,
        n_iter=iterations,
        momentum=0.99,
        # Let Griffin-Lim infer the frame count from the 128-column input;
        # crop/pad afterward to the repository's fixed waveform length.
        length=None,
        rand_init=False,
    ).detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.shape[0] < config.num_samples:
        waveform = np.pad(waveform, (0, config.num_samples - waveform.shape[0]))
    else:
        waveform = waveform[: config.num_samples]
    if not np.isfinite(waveform).all():
        raise ValueError("Decoded waveform contains NaN or infinity")
    peak = float(np.max(np.abs(waveform)))
    if peak > 1e-8:
        waveform = waveform * (target_peak / peak)
    return np.clip(waveform, -1.0, 1.0)


def write_waveform(path: Path, waveform: np.ndarray, config: SpectrogramConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, config.sample_rate, subtype="PCM_16")
