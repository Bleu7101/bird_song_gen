from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class VocoderSpectrogramConfig:
    """The fixed mel contract for nvidia/bigvgan_v2_22khz_80band_256x."""

    sample_rate: int = 22_050
    num_samples: int = 65_536
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = None
    expected_frames: int = 256
    model_id: str = "nvidia/bigvgan_v2_22khz_80band_256x"

    def __post_init__(self) -> None:
        for name in ("sample_rate", "num_samples", "n_fft", "hop_length", "win_length", "n_mels", "expected_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        if self.n_fft <= self.hop_length:
            raise ValueError("n_fft must exceed hop_length")
        if self.f_min < 0:
            raise ValueError("f_min cannot be negative")
        nyquist = self.sample_rate / 2.0
        if self.f_max is not None and not self.f_min < self.f_max <= nyquist:
            raise ValueError(f"f_max must be in ({self.f_min}, {nyquist}]")
        if self.frame_count(self.num_samples) != self.expected_frames:
            raise ValueError(
                f"num_samples={self.num_samples} produces {self.frame_count(self.num_samples)} frames, "
                f"not expected_frames={self.expected_frames}"
            )

    @property
    def padding(self) -> int:
        return (self.n_fft - self.hop_length) // 2

    def frame_count(self, num_samples: int) -> int:
        padded = num_samples + 2 * self.padding
        return 1 + (padded - self.n_fft) // self.hop_length

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VocoderSpectrogramConfig":
        return cls(**values)

    @classmethod
    def from_json(cls, path: Path) -> "VocoderSpectrogramConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class VocoderMelScaler:
    """Global training-only min/max scaling to the WGAN's [-1, 1] range."""

    minimum: float
    maximum: float
    count: int
    fitted_split: str = "train"

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError("scaler bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("scaler maximum must exceed minimum")
        if self.count < 1:
            raise ValueError("scaler count must be positive")

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return ((values - self.minimum) / self.span * 2.0 - 1.0).clamp(-1.0, 1.0)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values + 1.0) * 0.5 * self.span + self.minimum

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "count": self.count,
            "fitted_split": self.fitted_split,
            "normalization": "global_train_minmax_to_minus_one_one",
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VocoderMelScaler":
        return cls(
            minimum=float(values["minimum"]),
            maximum=float(values["maximum"]),
            count=int(values["count"]),
            fitted_split=str(values.get("fitted_split", "train")),
        )

    @classmethod
    def from_json(cls, path: Path) -> "VocoderMelScaler":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_vocoder_waveform(path: Path, config: VocoderSpectrogramConfig) -> torch.Tensor:
    """Load, resample, center-crop/pad, and volume-normalize one clip."""

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy()).mean(dim=0)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid waveform: {path}")
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)
    difference = waveform.numel() - config.num_samples
    if difference < 0:
        padding = -difference
        waveform = F.pad(waveform, (padding // 2, padding - padding // 2))
    elif difference > 0:
        start = difference // 2
        waveform = waveform[start : start + config.num_samples]
    if waveform.numel() != config.num_samples:
        raise RuntimeError(f"Expected {config.num_samples} samples, got {waveform.numel()}")
    peak = float(waveform.abs().amax())
    if peak > 1e-8:
        waveform = waveform * (0.95 / peak)
    return waveform.clamp(-1.0, 1.0).float()


_MEL_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_WINDOW_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_PINV_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def _hz_to_mel_slaney(frequencies: np.ndarray | float) -> np.ndarray:
    values = np.asanyarray(frequencies, dtype=np.float64)
    mels = values / (200.0 / 3.0)
    minimum_log_hz = 1_000.0
    minimum_log_mel = minimum_log_hz / (200.0 / 3.0)
    log_step = np.log(6.4) / 27.0
    return np.where(
        values >= minimum_log_hz,
        minimum_log_mel + np.log(np.maximum(values, minimum_log_hz) / minimum_log_hz) / log_step,
        mels,
    )


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    values = np.asanyarray(mels, dtype=np.float64)
    frequencies = (200.0 / 3.0) * values
    minimum_log_hz = 1_000.0
    minimum_log_mel = minimum_log_hz / (200.0 / 3.0)
    log_step = np.log(6.4) / 27.0
    return np.where(
        values >= minimum_log_mel,
        minimum_log_hz * np.exp(log_step * (values - minimum_log_mel)),
        frequencies,
    )


def _bigvgan_mel_basis(config: VocoderSpectrogramConfig) -> np.ndarray:
    """Dependency-light transcription of librosa's default Slaney basis."""

    f_max = config.sample_rate / 2 if config.f_max is None else config.f_max
    fft_frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    mel_points = np.linspace(
        float(_hz_to_mel_slaney(config.f_min)),
        float(_hz_to_mel_slaney(f_max)),
        config.n_mels + 2,
    )
    mel_frequencies = _mel_to_hz_slaney(mel_points)
    frequency_differences = np.diff(mel_frequencies)
    ramps = np.subtract.outer(mel_frequencies, fft_frequencies)
    weights = np.zeros((config.n_mels, 1 + config.n_fft // 2), dtype=np.float32)
    for index in range(config.n_mels):
        lower = -ramps[index] / frequency_differences[index]
        upper = ramps[index + 2] / frequency_differences[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))
    energy_normalization = 2.0 / (
        mel_frequencies[2 : config.n_mels + 2] - mel_frequencies[: config.n_mels]
    )
    weights *= energy_normalization[:, None]
    return weights


def _cache_key(waveform: torch.Tensor, config: VocoderSpectrogramConfig) -> tuple[Any, ...]:
    return (
        config.sample_rate,
        config.n_fft,
        config.hop_length,
        config.win_length,
        config.n_mels,
        config.f_min,
        config.f_max,
        waveform.device.type,
        waveform.device.index,
        waveform.dtype,
    )


def _mel_basis(waveform: torch.Tensor, config: VocoderSpectrogramConfig) -> torch.Tensor:
    key = _cache_key(waveform, config)
    if key not in _MEL_CACHE:
        basis = _bigvgan_mel_basis(config)
        _MEL_CACHE[key] = torch.from_numpy(basis).to(device=waveform.device, dtype=waveform.dtype)
        _WINDOW_CACHE[key] = torch.hann_window(config.win_length, device=waveform.device, dtype=waveform.dtype)
    return _MEL_CACHE[key]


def waveform_to_vocoder_mel(waveform: torch.Tensor, config: VocoderSpectrogramConfig) -> torch.Tensor:
    """Reproduce BigVGAN's official magnitude log-mel frontend."""

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] != config.num_samples:
        raise ValueError(f"Expected [samples] or [batch, {config.num_samples}], got {tuple(waveform.shape)}")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or infinity")
    if float(waveform.amin()) < -1.0001 or float(waveform.amax()) > 1.0001:
        raise ValueError("waveform values must be in [-1, 1]")
    key = _cache_key(waveform, config)
    _mel_basis(waveform, config)
    padded = F.pad(waveform.unsqueeze(1), (config.padding, config.padding), mode="reflect").squeeze(1)
    spectrum = torch.stft(
        padded,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=_WINDOW_CACHE[key],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-9)
    logmel = torch.log(torch.clamp(torch.matmul(_MEL_CACHE[key], magnitude), min=1e-5))
    expected = (waveform.shape[0], config.n_mels, config.expected_frames)
    if tuple(logmel.shape) != expected:
        raise RuntimeError(f"Expected mel shape {expected}, got {tuple(logmel.shape)}")
    return logmel


def _validate_logmel(values: torch.Tensor, config: VocoderSpectrogramConfig) -> torch.Tensor:
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    expected = (config.n_mels, config.expected_frames)
    if values.ndim != 2 or tuple(values.shape) != expected:
        raise ValueError(f"Expected raw log-mel {expected}, got {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise ValueError("raw log-mel contains NaN or infinity")
    return values


def griffin_lim_from_vocoder_mel(
    raw_logmel: torch.Tensor,
    config: VocoderSpectrogramConfig,
    iterations: int = 32,
) -> torch.Tensor:
    """Decode the same mel contract with Griffin-Lim for a paired baseline."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    values = _validate_logmel(raw_logmel, config).detach()
    basis = _mel_basis(values, config)
    key = _cache_key(values, config)
    if key not in _PINV_CACHE:
        _PINV_CACHE[key] = torch.linalg.pinv(basis)
    linear_magnitude = (_PINV_CACHE[key] @ values.exp()).clamp_min(0.0)
    # ``torchaudio.functional.griffinlim`` always rebuilds a centered STFT.
    # BigVGAN's frontend instead reflect-pads by ``(n_fft-hop)/2`` and uses
    # ``center=False``; that produces exactly 256 frames for this contract.
    # Use a small overlap-add implementation so the baseline has the same
    # rectangular frame geometry rather than silently introducing a 257th
    # frame.
    window = _WINDOW_CACHE[key]
    padded_length = config.num_samples + 2 * config.padding
    frame_count = config.expected_frames
    denominator = F.fold(
        window.square().expand(frame_count, -1).transpose(0, 1).reshape(1, config.n_fft, frame_count),
        output_size=(1, padded_length),
        kernel_size=(1, config.n_fft),
        stride=(1, config.hop_length),
    ).reshape(-1)
    angles = torch.ones_like(linear_magnitude, dtype=torch.complex64)
    for _ in range(iterations):
        frames = torch.fft.irfft((linear_magnitude * angles).transpose(0, 1), n=config.n_fft)
        frames = frames * window
        overlap_added = F.fold(
            frames.transpose(0, 1).reshape(1, config.n_fft, frame_count),
            output_size=(1, padded_length),
            kernel_size=(1, config.n_fft),
            stride=(1, config.hop_length),
        ).reshape(-1)
        padded_waveform = overlap_added / denominator.clamp_min(1e-8)
        rebuilt = torch.stft(
            padded_waveform.unsqueeze(0),
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )[0]
        angles = rebuilt / rebuilt.abs().clamp_min(1e-8)
    frames = torch.fft.irfft((linear_magnitude * angles).transpose(0, 1), n=config.n_fft) * window
    waveform = F.fold(
        frames.transpose(0, 1).reshape(1, config.n_fft, frame_count),
        output_size=(1, padded_length),
        kernel_size=(1, config.n_fft),
        stride=(1, config.hop_length),
    ).reshape(-1) / denominator.clamp_min(1e-8)
    waveform = waveform[config.padding : config.padding + config.num_samples]
    if waveform.numel() < config.num_samples:
        waveform = F.pad(waveform, (0, config.num_samples - waveform.numel()))
    waveform = waveform[: config.num_samples]
    peak = float(waveform.abs().amax())
    if peak > 1.0:
        waveform = waveform / peak
    return waveform.float().cpu()


def load_bigvgan(source_dir: Path, device: torch.device, config: VocoderSpectrogramConfig) -> torch.nn.Module:
    """Load the official local BigVGAN source/checkpoint without its CUDA extension."""

    source_dir = source_dir.resolve()
    required = (source_dir / "bigvgan.py", source_dir / "config.json", source_dir / "bigvgan_generator.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BigVGAN snapshot is missing: {missing}")
    source_string = str(source_dir)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    previous_utils = sys.modules.get("utils")
    shim = types.ModuleType("utils")

    def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
        if "Conv" in module.__class__.__name__:
            module.weight.data.normal_(mean, std)

    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)

    shim.init_weights = init_weights  # type: ignore[attr-defined]
    shim.get_padding = get_padding  # type: ignore[attr-defined]
    sys.modules["utils"] = shim
    try:
        module = importlib.import_module("bigvgan")
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    construction_device = device if device.type == "cuda" else torch.device("cpu")
    model = module.BigVGAN._from_pretrained(
        model_id=str(source_dir),
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=True,
        token=None,
        use_cuda_kernel=False,
        map_location=str(construction_device),
    )
    hyperparameters = model.h
    expected = {
        "sampling_rate": config.sample_rate,
        "segment_size": config.num_samples,
        "n_fft": config.n_fft,
        "hop_size": config.hop_length,
        "win_size": config.win_length,
        "num_mels": config.n_mels,
        "fmin": config.f_min,
        "fmax": config.f_max,
    }
    mismatches = {}
    for name, value in expected.items():
        actual = hyperparameters.get(name) if hasattr(hyperparameters, "get") else getattr(hyperparameters, name)
        if actual != value:
            mismatches[name] = {"expected": value, "checkpoint": actual}
    if mismatches:
        raise ValueError(f"BigVGAN checkpoint does not match the mel contract: {mismatches}")
    model.remove_weight_norm()
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def vocoder_mel_to_waveform(
    raw_logmel: torch.Tensor,
    model: torch.nn.Module,
    config: VocoderSpectrogramConfig,
) -> torch.Tensor:
    single = raw_logmel.ndim == 2
    if single:
        raw_logmel = raw_logmel.unsqueeze(0)
    expected = (config.n_mels, config.expected_frames)
    if raw_logmel.ndim != 3 or tuple(raw_logmel.shape[1:]) != expected:
        raise ValueError(f"Expected [batch, {expected[0]}, {expected[1]}], got {tuple(raw_logmel.shape)}")
    if not torch.isfinite(raw_logmel).all():
        raise ValueError("raw log-mel contains NaN or infinity")
    device = next(model.parameters()).device
    waveform = model(raw_logmel.to(device)).detach().float().cpu()
    if waveform.ndim != 3 or waveform.shape[1:] != (1, config.num_samples):
        raise RuntimeError(f"Expected BigVGAN output [batch, 1, {config.num_samples}], got {tuple(waveform.shape)}")
    waveform = waveform[:, 0].clamp(-1.0, 1.0)
    if not torch.isfinite(waveform).all():
        raise RuntimeError("BigVGAN produced NaN or infinity")
    return waveform[0] if single else waveform
