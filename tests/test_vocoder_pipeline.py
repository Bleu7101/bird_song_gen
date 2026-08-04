from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bird_song.vocoder import (
    VocoderMelNormalizer,
    VocoderSpectrogramConfig,
    _bigvgan_mel_basis,
    vocoder_mel_to_waveform,
    waveform_to_vocoder_mel,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(1)


def official_frontend_reference(
    waveform: torch.Tensor,
    config: VocoderSpectrogramConfig,
) -> torch.Tensor:
    """Independent transcription of BigVGAN's official meldataset helper."""

    mel_basis = torch.from_numpy(_bigvgan_mel_basis(config)).to(
        device=waveform.device, dtype=waveform.dtype
    )
    window = torch.hann_window(config.win_length, device=waveform.device, dtype=waveform.dtype)
    padded = F.pad(
        waveform[None, None],
        (config.padding, config.padding),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        padded,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-9)
    return torch.log(torch.clamp(mel_basis @ magnitude, min=1e-5))


def test_vocoder_contract_and_exact_frontend_parity() -> None:
    production_config = VocoderSpectrogramConfig()
    assert production_config.frame_count(production_config.num_samples) == 256
    assert (production_config.n_mels, production_config.expected_frames) == (80, 256)

    config = VocoderSpectrogramConfig(
        sample_rate=8_000,
        num_samples=1_024,
        n_fft=256,
        hop_length=64,
        win_length=256,
        n_mels=20,
        f_max=4_000,
        expected_frames=16,
        model_id="test/frontend",
    )
    generator = torch.Generator().manual_seed(42)
    waveform = 0.2 * torch.randn(config.num_samples, generator=generator).to(DEVICE)

    actual = waveform_to_vocoder_mel(waveform, config)
    expected = official_frontend_reference(waveform, config)

    assert actual.shape == (1, 20, 16)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_exact_parity_against_downloaded_bigvgan_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "bigvgan_v2_22khz_80band_256x"
    )
    if not (source / "meldataset.py").is_file():
        pytest.skip("Official BigVGAN source has not been fetched")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.syspath_prepend(str(source))
    from librosa.filters import mel as official_mel_basis

    lightweight_util = types.ModuleType("librosa.util")
    lightweight_util.normalize = lambda values, *args, **kwargs: values  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "librosa.util", lightweight_util)
    config = VocoderSpectrogramConfig(
        sample_rate=8_000,
        num_samples=1_024,
        n_fft=256,
        hop_length=64,
        win_length=256,
        n_mels=20,
        f_max=4_000,
        expected_frames=16,
        model_id="test/frontend",
    )
    official = importlib.import_module("meldataset")
    official.librosa_mel_fn = official_mel_basis
    official.mel_basis_cache.clear()
    official.hann_window_cache.clear()
    generator = torch.Generator().manual_seed(123)
    waveform = (0.1 * torch.randn(config.num_samples, generator=generator)).to(DEVICE)
    actual = waveform_to_vocoder_mel(waveform, config)
    expected = official.mel_spectrogram(
        waveform[None],
        config.n_fft,
        config.n_mels,
        config.sample_rate,
        config.hop_length,
        config.win_length,
        config.f_min,
        config.f_max,
        center=False,
    )
    assert actual.shape == (1, 20, 16)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_normalizer_round_trip_is_exact_enough() -> None:
    normalizer = VocoderMelNormalizer(mean=-4.25, std=1.75, count=1_000)
    values = torch.linspace(-12.0, 2.0, 80 * 256).reshape(1, 80, 256)
    restored = normalizer.denormalize(normalizer.normalize(values))
    torch.testing.assert_close(restored, values, rtol=1e-6, atol=1e-6)


def test_vocoder_interface_returns_valid_length() -> None:
    config = VocoderSpectrogramConfig(
        sample_rate=8_000,
        num_samples=1_024,
        n_fft=256,
        hop_length=64,
        win_length=256,
        n_mels=20,
        f_max=4_000,
        expected_frames=16,
        model_id="test/frontend",
    )
    raw_logmel = torch.zeros(config.n_mels, config.expected_frames)

    class FakeVocoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                inputs.shape[0],
                1,
                config.num_samples,
                device=inputs.device,
            ) + self.anchor

    decoded = vocoder_mel_to_waveform(raw_logmel, FakeVocoder().to(DEVICE), config)
    assert decoded.shape == (config.num_samples,)
    assert torch.isfinite(decoded).all()
    assert float(decoded.abs().max()) <= 1.0
