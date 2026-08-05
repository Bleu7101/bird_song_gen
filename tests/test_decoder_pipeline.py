from __future__ import annotations

import io

import torch
from torch import nn
import numpy as np

from bird_song.generation.wgan_gp import ConditionalCritic, ConditionalGenerator, WGANConfig
from bird_song.vocoder import (
    VocoderMelScaler,
    VocoderSpectrogramConfig,
    vocoder_mel_to_waveform,
    waveform_to_vocoder_mel,
)


DEVICE = torch.device("cpu")
torch.set_num_threads(1)


def small_config() -> VocoderSpectrogramConfig:
    return VocoderSpectrogramConfig(
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


def test_production_contract_is_80_by_256() -> None:
    config = VocoderSpectrogramConfig()
    assert config.frame_count(config.num_samples) == 256
    assert (config.n_mels, config.expected_frames) == (80, 256)
    assert config.num_samples == 65_536


def test_frontend_shape_and_finiteness() -> None:
    config = small_config()
    generator = torch.Generator().manual_seed(42)
    waveform = 0.2 * torch.randn(config.num_samples, generator=generator)
    mel = waveform_to_vocoder_mel(waveform, config)
    assert mel.shape == (1, config.n_mels, config.expected_frames)
    assert torch.isfinite(mel).all()


def test_frontend_matches_independent_official_formula() -> None:
    config = small_config()
    generator = torch.Generator().manual_seed(123)
    waveform = 0.1 * torch.randn(config.num_samples, generator=generator)
    actual = waveform_to_vocoder_mel(waveform, config)[0]
    import torch.nn.functional as F

    def hz_to_mel(values: np.ndarray | float) -> np.ndarray:
        values = np.asanyarray(values, dtype=np.float64)
        linear = values / (200.0 / 3.0)
        minimum_log_hz = 1_000.0
        minimum_log_mel = minimum_log_hz / (200.0 / 3.0)
        log_step = np.log(6.4) / 27.0
        return np.where(values >= minimum_log_hz, minimum_log_mel + np.log(np.maximum(values, minimum_log_hz) / minimum_log_hz) / log_step, linear)

    def mel_to_hz(values: np.ndarray) -> np.ndarray:
        values = np.asanyarray(values, dtype=np.float64)
        linear = (200.0 / 3.0) * values
        minimum_log_hz = 1_000.0
        minimum_log_mel = minimum_log_hz / (200.0 / 3.0)
        log_step = np.log(6.4) / 27.0
        return np.where(values >= minimum_log_mel, minimum_log_hz * np.exp(log_step * (values - minimum_log_mel)), linear)

    fft_frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    mel_points = np.linspace(float(hz_to_mel(config.f_min)), float(hz_to_mel(config.f_max)), config.n_mels + 2)
    mel_frequencies = mel_to_hz(mel_points)
    differences = np.diff(mel_frequencies)
    ramps = np.subtract.outer(mel_frequencies, fft_frequencies)
    basis_np = np.zeros((config.n_mels, 1 + config.n_fft // 2), dtype=np.float32)
    for index in range(config.n_mels):
        basis_np[index] = np.maximum(0.0, np.minimum(-ramps[index] / differences[index], ramps[index + 2] / differences[index + 1]))
    basis_np *= (2.0 / (mel_frequencies[2 : config.n_mels + 2] - mel_frequencies[: config.n_mels]))[:, None]
    basis = torch.from_numpy(basis_np)
    padded = F.pad(waveform[None, None], (config.padding, config.padding), mode="reflect").squeeze(1)
    spectrum = torch.stft(
        padded,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=torch.hann_window(config.win_length),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    expected = torch.log(torch.clamp(basis @ torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-9), min=1e-5))[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_scaler_round_trip() -> None:
    scaler = VocoderMelScaler(minimum=-11.5, maximum=3.0, count=100)
    values = torch.linspace(-11.5, 3.0, 80 * 256).reshape(1, 80, 256)
    torch.testing.assert_close(scaler.denormalize(scaler.normalize(values)), values, rtol=1e-6, atol=1e-6)


def test_rectangular_wgan_shapes_and_checkpoint_reload() -> None:
    config = WGANConfig(base_channels=8, critic_base_channels=8)
    generator = ConditionalGenerator(config).to(DEVICE)
    critic = ConditionalCritic(config).to(DEVICE)
    labels = torch.tensor([0, 1], device=DEVICE)
    noise = torch.randn(2, config.latent_dim, device=DEVICE)
    fake = generator(noise, labels)
    assert fake.shape == (2, 1, 80, 256)
    assert critic(fake, labels).shape == (2,)
    buffer = io.BytesIO()
    torch.save(generator.state_dict(), buffer)
    buffer.seek(0)
    reloaded = ConditionalGenerator(config).to(DEVICE)
    reloaded.load_state_dict(torch.load(buffer, weights_only=True))
    assert reloaded(noise, labels).shape == fake.shape


def test_decoder_interface_validates_output_length() -> None:
    config = small_config()

    class FakeVocoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            return torch.zeros(mel.shape[0], 1, config.num_samples, device=mel.device) + self.anchor

    decoded = vocoder_mel_to_waveform(torch.zeros(config.n_mels, config.expected_frames), FakeVocoder(), config)
    assert decoded.shape == (config.num_samples,)
    assert torch.isfinite(decoded).all()
    assert float(decoded.abs().max()) <= 1.0
