import torch
import numpy as np

from bird_song.config import SpectrogramConfig
from bird_song.generation.audio_decode import normalized_logmel_to_waveform
from bird_song.generation.diffusion import (
    ConditionalLatentDenoiser,
    LatentDiffusionConfig,
    diffusion_loss,
    sample_latents,
)
from bird_song.generation.evaluation import detail_metrics, diversity_metrics, validate_generated
from bird_song.generation.token_transformer import ConditionalTokenTransformer, TokenTransformerConfig
from bird_song.generation.vqgan import ConditionalVQGAN, PatchDiscriminator, VQGANConfig
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


def test_vqgan_round_trip_and_tokens() -> None:
    config = VQGANConfig(base_channels=16, discriminator_base_channels=16, latent_channels=32, codebook_size=64)
    model = ConditionalVQGAN(config)
    discriminator = PatchDiscriminator(config)
    real = torch.randn(2, 1, 128, 128).clamp(-1, 1)
    labels = torch.tensor([0, 1])
    reconstruction, indices, latents, loss = model(real, labels)
    assert reconstruction.shape == real.shape
    assert indices.shape == (2, config.latent_grid, config.latent_grid)
    assert latents.shape == (2, config.latent_channels, config.latent_grid, config.latent_grid)
    assert discriminator(reconstruction).ndim == 4
    assert torch.isfinite(loss)
    assert model.decode(indices, labels).shape == real.shape


def test_token_transformer_and_diffusion_contracts() -> None:
    vq_config = VQGANConfig(base_channels=16, discriminator_base_channels=16, latent_channels=32, codebook_size=64)
    token_config = TokenTransformerConfig(token_grid=vq_config.latent_grid, codebook_size=vq_config.codebook_size, d_model=32, num_heads=4, num_layers=1, feedforward_dim=64)
    tokens = torch.randint(vq_config.codebook_size, (2, token_config.token_grid, token_config.token_grid))
    labels = torch.tensor([0, 2])
    transformer = ConditionalTokenTransformer(token_config)
    assert torch.isfinite(transformer.loss(tokens, labels))
    assert transformer.generate(labels, temperature=0).shape == tokens.shape

    diffusion_config = LatentDiffusionConfig(latent_channels=32, latent_grid=vq_config.latent_grid, time_embedding_dim=32, base_channels=32, diffusion_steps=10)
    denoiser = ConditionalLatentDenoiser(diffusion_config)
    latents = torch.randn(2, 32, vq_config.latent_grid, vq_config.latent_grid)
    assert torch.isfinite(diffusion_loss(denoiser, latents, labels))
    assert sample_latents(denoiser, labels).shape == latents.shape


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
