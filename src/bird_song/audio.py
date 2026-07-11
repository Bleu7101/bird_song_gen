from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .config import SpectrogramConfig


def load_waveform(path: Path, config: SpectrogramConfig, training: bool) -> torch.Tensor:
    """Load a file as mono float32 audio, resample it, and crop/pad to a fixed size."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy()).mean(dim=0)
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Audio file contains NaN or infinity: {path}")

    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)

    target = config.num_samples
    if waveform.numel() < target:
        total_padding = target - waveform.numel()
        left = torch.randint(total_padding + 1, ()).item() if training else total_padding // 2
        waveform = F.pad(waveform, (left, total_padding - left))
    elif waveform.numel() > target:
        available = waveform.numel() - target
        start = torch.randint(available + 1, ()).item() if training else available // 2
        waveform = waveform[start : start + target]

    if training:
        gain = torch.empty(()).uniform_(0.75, 1.25)
        waveform = waveform * gain
        if torch.rand(()) < 0.35:
            noise_scale = waveform.square().mean().sqrt().clamp_min(1e-4)
            waveform = waveform + torch.randn_like(waveform) * noise_scale * torch.empty(()).uniform_(0.002, 0.02)

    return waveform.clamp(-1.0, 1.0)


def normalize_logmel(logmel_db: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    """Map a relative dB spectrogram to roughly [-1, 1]."""
    relative = logmel_db - logmel_db.amax(dim=(-2, -1), keepdim=True)
    relative = relative.clamp(min=-top_db, max=0.0)
    return relative.mul(2.0 / top_db).add(1.0)


class LogMelTransform(torch.nn.Module):
    def __init__(self, config: SpectrogramConfig, training: bool = False) -> None:
        super().__init__()
        self.config = config
        self.training_mode = training
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=config.top_db)
        self.frequency_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=16)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.to_db(self.mel(waveform))
        width = self.config.spectrogram_width
        if spec.shape[-1] < width:
            spec = F.pad(spec, (0, width - spec.shape[-1]), value=float(spec.amin()))
        elif spec.shape[-1] > width:
            start = (
                torch.randint(spec.shape[-1] - width + 1, ()).item()
                if self.training_mode
                else (spec.shape[-1] - width) // 2
            )
            spec = spec[..., start : start + width]

        spec = normalize_logmel(spec, self.config.top_db)
        if self.training_mode:
            # Normalized -1 corresponds to the spectrogram floor (silence).
            # TorchAudio's default mask value is 0, which would inject artificial
            # mid-level energy into this [-1, 1] representation.
            if torch.rand(()) < 0.5:
                spec = self.frequency_mask(spec, mask_value=-1.0)
            if torch.rand(()) < 0.5:
                spec = self.time_mask(spec, mask_value=-1.0)
        return spec.unsqueeze(0)


def load_generated_spectrogram(path: Path, config: SpectrogramConfig) -> torch.Tensor:
    """Load a generated 2-D spectrogram and convert common ranges to [-1, 1]."""
    array = np.load(path, allow_pickle=False)
    spec = torch.as_tensor(array, dtype=torch.float32).squeeze()
    if spec.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram in {path}, got shape {tuple(spec.shape)}")
    if not torch.isfinite(spec).all():
        raise ValueError(f"Spectrogram contains NaN or infinity: {path}")

    spec = F.interpolate(
        spec[None, None],
        size=(config.n_mels, config.spectrogram_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    minimum, maximum = float(spec.amin()), float(spec.amax())
    if minimum >= -1.05 and maximum <= 1.05:
        if minimum >= 0.0:
            spec = spec.mul(2.0).sub(1.0)
    else:
        spec = normalize_logmel(spec, config.top_db)
    return spec.clamp(-1.0, 1.0).unsqueeze(0)
