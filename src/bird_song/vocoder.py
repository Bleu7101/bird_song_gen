from __future__ import annotations

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
    """Exact log-mel contract used by a frozen BigVGAN checkpoint."""

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
        integer_fields = {
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_mels": self.n_mels,
            "expected_frames": self.expected_frames,
        }
        invalid = [name for name, value in integer_fields.items() if value < 1]
        if invalid:
            raise ValueError(f"Vocoder config fields must be positive: {invalid}")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        if self.n_fft <= self.hop_length:
            raise ValueError("n_fft must exceed hop_length for BigVGAN reflect padding")
        if self.f_min < 0:
            raise ValueError("f_min cannot be negative")
        nyquist = self.sample_rate / 2
        if self.f_max is not None and not self.f_min < self.f_max <= nyquist:
            raise ValueError(f"f_max must be in ({self.f_min}, {nyquist}]")
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        actual_frames = self.frame_count(self.num_samples)
        if actual_frames != self.expected_frames:
            raise ValueError(
                f"num_samples={self.num_samples} produces {actual_frames} frames, "
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
class VocoderMelNormalizer:
    """Invertible scalar standardization fitted on training log-mel bins only."""

    mean: float
    std: float
    count: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("Normalizer mean must be finite")
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("Normalizer std must be finite and positive")
        if self.count < 1:
            raise ValueError("Normalizer count must be positive")

    def normalize(self, raw_logmel: torch.Tensor) -> torch.Tensor:
        return (raw_logmel - self.mean) / self.std

    def denormalize(self, normalized_logmel: torch.Tensor) -> torch.Tensor:
        return normalized_logmel * self.std + self.mean

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "fitted_split": "train",
            "normalization": "global_scalar_zscore",
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VocoderMelNormalizer":
        return cls(mean=float(values["mean"]), std=float(values["std"]), count=int(values["count"]))

    @classmethod
    def from_json(cls, path: Path) -> "VocoderMelNormalizer":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_vocoder_waveform(
    path: Path,
    config: VocoderSpectrogramConfig,
    *,
    training: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Load mono audio and crop/pad it to the frozen vocoder segment length."""

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy()).mean(dim=0)
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Audio contains NaN or infinity: {path}")
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)

    difference = waveform.numel() - config.num_samples
    if difference < 0:
        total_padding = -difference
        left = total_padding // 2
        waveform = F.pad(waveform, (left, total_padding - left))
    elif difference > 0:
        if training:
            start = int(torch.randint(difference + 1, (), generator=generator))
        else:
            start = difference // 2
        waveform = waveform[start : start + config.num_samples]
    waveform = waveform.clamp(-1.0, 1.0)
    if waveform.shape != (config.num_samples,):
        raise RuntimeError(f"Expected {config.num_samples} samples, got {tuple(waveform.shape)}")
    return waveform


_MEL_BASIS_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_HANN_WINDOW_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_MEL_PSEUDOINVERSE_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


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
    """Direct, dependency-light transcription of librosa's default Slaney mel basis."""

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


def waveform_to_vocoder_mel(
    waveform: torch.Tensor,
    config: VocoderSpectrogramConfig,
) -> torch.Tensor:
    """Reproduce BigVGAN's official magnitude log-mel frontend exactly."""

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [samples] or [batch, samples], got {tuple(waveform.shape)}")
    if waveform.shape[-1] != config.num_samples:
        raise ValueError(f"Expected {config.num_samples} samples, got {waveform.shape[-1]}")
    if not torch.isfinite(waveform).all():
        raise ValueError("Waveform contains NaN or infinity")
    if float(waveform.amin()) < -1.0001 or float(waveform.amax()) > 1.0001:
        raise ValueError("Waveform values must be in [-1, 1]")

    device_key = (waveform.device.type, waveform.device.index, waveform.dtype)
    basis_key = (
        config.n_fft,
        config.n_mels,
        config.sample_rate,
        config.hop_length,
        config.win_length,
        config.f_min,
        config.f_max,
        *device_key,
    )
    if basis_key not in _MEL_BASIS_CACHE:
        mel = _bigvgan_mel_basis(config)
        _MEL_BASIS_CACHE[basis_key] = torch.from_numpy(mel).to(
            device=waveform.device, dtype=waveform.dtype
        )
        _HANN_WINDOW_CACHE[basis_key] = torch.hann_window(
            config.win_length, device=waveform.device, dtype=waveform.dtype
        )

    padded = F.pad(
        waveform.unsqueeze(1),
        (config.padding, config.padding),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        padded,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=_HANN_WINDOW_CACHE[basis_key],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-9)
    mel = torch.matmul(_MEL_BASIS_CACHE[basis_key], magnitude)
    logmel = torch.log(torch.clamp(mel, min=1e-5))
    expected = (waveform.shape[0], config.n_mels, config.expected_frames)
    if tuple(logmel.shape) != expected:
        raise RuntimeError(f"Expected vocoder mel shape {expected}, got {tuple(logmel.shape)}")
    return logmel


def griffin_lim_from_vocoder_mel(
    raw_logmel: torch.Tensor,
    config: VocoderSpectrogramConfig,
    *,
    iterations: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Approximate the frozen vocoder mel contract with a Griffin-Lim baseline."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    mel = _validate_logmel(raw_logmel, config).detach()
    inverse_key = (
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.f_min,
        config.f_max,
        mel.device.type,
        mel.device.index,
        mel.dtype,
    )
    if inverse_key not in _MEL_PSEUDOINVERSE_CACHE:
        basis = torch.from_numpy(_bigvgan_mel_basis(config)).to(
            device=mel.device, dtype=mel.dtype
        )
        _MEL_PSEUDOINVERSE_CACHE[inverse_key] = torch.linalg.pinv(basis)
    linear_magnitude = (
        (_MEL_PSEUDOINVERSE_CACHE[inverse_key] @ mel.exp())
        .clamp_min(0.0)
    )
    griffin_length = (config.expected_frames - 1) * config.hop_length
    random_devices = [mel.device] if mel.device.type == "cuda" else []
    with torch.random.fork_rng(devices=random_devices):
        torch.manual_seed(seed)
        waveform = torchaudio.functional.griffinlim(
        linear_magnitude,
            torch.hann_window(config.win_length, device=mel.device, dtype=mel.dtype),
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            power=1.0,
            n_iter=iterations,
            momentum=0.99,
            length=griffin_length,
            rand_init=True,
        )
    missing = config.num_samples - waveform.numel()
    if missing > 0:
        waveform = F.pad(waveform, (missing // 2, missing - missing // 2))
    elif missing < 0:
        start = (-missing) // 2
        waveform = waveform[start : start + config.num_samples]
    peak = float(waveform.abs().amax())
    if peak > 1.0:
        waveform = waveform / peak
    result = waveform.float().cpu()
    if result.shape != (config.num_samples,) or not torch.isfinite(result).all():
        raise RuntimeError("Griffin-Lim produced an invalid waveform")
    return result


def _validate_logmel(raw_logmel: torch.Tensor, config: VocoderSpectrogramConfig) -> torch.Tensor:
    if raw_logmel.ndim == 3 and raw_logmel.shape[0] == 1:
        raw_logmel = raw_logmel[0]
    expected = (config.n_mels, config.expected_frames)
    if raw_logmel.ndim != 2 or tuple(raw_logmel.shape) != expected:
        raise ValueError(f"Expected raw log-mel shape {expected} or (1, {expected}), got {tuple(raw_logmel.shape)}")
    if not torch.isfinite(raw_logmel).all():
        raise ValueError("Raw log-mel contains NaN or infinity")
    return raw_logmel


def load_bigvgan(
    model_id_or_path: str | Path,
    device: torch.device,
    config: VocoderSpectrogramConfig,
    *,
    source_dir: Path | None = None,
) -> torch.nn.Module:
    """Load and freeze the official BigVGAN implementation without its CUDA extension."""

    if source_dir is not None:
        source_dir = source_dir.resolve()
        if not (source_dir / "bigvgan.py").is_file():
            raise FileNotFoundError(f"Expected BigVGAN source at {source_dir / 'bigvgan.py'}")
        sys.path.insert(0, str(source_dir))
    training_utils = sys.modules.get("utils")
    inference_utils = types.ModuleType("utils")

    def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
        if "Conv" in module.__class__.__name__:
            module.weight.data.normal_(mean, std)

    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)

    inference_utils.init_weights = init_weights  # type: ignore[attr-defined]
    inference_utils.get_padding = get_padding  # type: ignore[attr-defined]
    sys.modules["utils"] = inference_utils
    print("Importing the official BigVGAN inference source...", flush=True)
    try:
        import bigvgan  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "BigVGAN source is unavailable. Run scripts/08_fetch_bigvgan.py and pass "
            "--bigvgan-source external/BigVGAN."
        ) from error
    finally:
        if training_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = training_utils

    construction_device = device if device.type == "cuda" else torch.device("cpu")
    print(f"Loading the frozen BigVGAN generator on {construction_device}...", flush=True)
    with torch.device(construction_device):
        model = bigvgan.BigVGAN._from_pretrained(
            model_id=str(model_id_or_path),
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=Path(model_id_or_path).is_dir(),
            token=None,
            use_cuda_kernel=False,
            map_location=str(construction_device),
        )
    print("Validating the BigVGAN checkpoint contract...", flush=True)
    model_config = model.h
    expected_values = {
        "sampling_rate": config.sample_rate,
        "segment_size": config.num_samples,
        "n_fft": config.n_fft,
        "hop_size": config.hop_length,
        "win_size": config.win_length,
        "num_mels": config.n_mels,
        "fmin": config.f_min,
        "fmax": config.f_max,
    }
    mismatches = {
        name: {"expected": expected, "checkpoint": model_config.get(name)}
        for name, expected in expected_values.items()
        if model_config.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"BigVGAN checkpoint does not match the requested mel contract: {mismatches}")
    print("Removing inference-time weight normalization...", flush=True)
    model.remove_weight_norm()
    model.eval().requires_grad_(False).to(device)
    return model


@torch.inference_mode()
def vocoder_mel_to_waveform(
    raw_logmel: torch.Tensor,
    model: torch.nn.Module,
    config: VocoderSpectrogramConfig,
) -> torch.Tensor:
    """Decode an exact-contract raw log-mel tensor with a frozen BigVGAN model."""

    single = raw_logmel.ndim == 2 or (raw_logmel.ndim == 3 and raw_logmel.shape[0] == 1)
    if raw_logmel.ndim == 2:
        raw_logmel = raw_logmel.unsqueeze(0)
    if raw_logmel.ndim != 3 or tuple(raw_logmel.shape[1:]) != (
        config.n_mels,
        config.expected_frames,
    ):
        raise ValueError(
            f"Expected [batch, {config.n_mels}, {config.expected_frames}], got {tuple(raw_logmel.shape)}"
        )
    if not torch.isfinite(raw_logmel).all():
        raise ValueError("Raw log-mel contains NaN or infinity")
    device = next(model.parameters()).device
    waveform = model(raw_logmel.to(device)).detach().float().cpu()
    if waveform.ndim != 3 or waveform.shape[1:] != (1, config.num_samples):
        raise RuntimeError(
            f"Expected BigVGAN output [batch, 1, {config.num_samples}], got {tuple(waveform.shape)}"
        )
    waveform = waveform[:, 0].clamp(-1.0, 1.0)
    if not torch.isfinite(waveform).all():
        raise RuntimeError("BigVGAN produced NaN or infinity")
    return waveform[0] if single else waveform
