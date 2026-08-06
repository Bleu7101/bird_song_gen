import torch
import numpy as np

from bird_song.config import SpectrogramConfig
from bird_song.generation.audio_decode import normalized_logmel_to_waveform
from bird_song.generation.evaluation import detail_metrics, diversity_metrics, validate_generated
from bird_song.generation.wgan_gp import ConditionalCritic, ConditionalGenerator, WGANConfig, gradient_penalty


def test_wgan_gp_contract_and_penalty() -> None:
    config = WGANConfig(base_channels=16, critic_base_channels=16)
    generator = ConditionalGenerator(config)
    critic = ConditionalCritic(config)
    labels = torch.tensor([0, 2])
    real = torch.randn(2, 1, 128, 128)
    fake = generator(torch.randn(2, config.latent_dim), labels)
    assert fake.shape == real.shape
    assert critic(fake, labels).shape == (2,)
    penalty = gradient_penalty(critic, real, fake, labels)
    assert torch.isfinite(penalty)


def test_generator_evaluation_contract() -> None:
    real = torch.randn(4, 1, 128, 128)
    generated = real * 0.5
    detail = detail_metrics(real, generated)
    diversity = diversity_metrics(generated)
    assert 0 < detail["time_detail_ratio"] < 1
    assert 0 < detail["frequency_detail_ratio"] < 1
    assert diversity["finite"] == 1.0
    assert validate_generated(generated)["shape"] == [4, 1, 128, 128]


def test_griffin_lim_decoder_returns_fixed_finite_waveform() -> None:
    config = SpectrogramConfig()
    waveform = normalized_logmel_to_waveform(np.full((config.n_mels, config.spectrogram_width), -1, dtype=np.float32), config, iterations=1)
    assert waveform.shape == (config.num_samples,)
    assert np.isfinite(waveform).all()
    assert float(np.max(np.abs(waveform))) <= 1.0
