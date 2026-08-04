from __future__ import annotations

import torch

from bird_song.vocoder_vae.losses import VocoderVAELossConfig, vae_loss
from bird_song.vocoder_vae.model import ConditionalVocoderVAE, VocoderVAEConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(1)


def test_spatial_vocoder_vae_shape_and_loss_contract() -> None:
    config = VocoderVAEConfig(
        latent_channels=1,
        class_embed_dim=2,
        base_channels=1,
    )
    model = ConditionalVocoderVAE(config).to(DEVICE)
    inputs = torch.randn(1, 1, 80, 256, device=DEVICE)
    labels = torch.tensor([2], device=DEVICE)

    reconstruction, mu, logvar = model(inputs, labels, sample_latent=False)
    assert config.latent_shape == (1, 5, 16)
    assert reconstruction.shape == inputs.shape
    assert mu.shape == (1, 1, 5, 16)
    assert logvar.shape == mu.shape

    loss, parts = vae_loss(
        reconstruction,
        inputs,
        mu,
        logvar,
        beta=1e-3,
        config=VocoderVAELossConfig(),
    )
    assert set(parts) == {
        "loss",
        "recon",
        "mse",
        "mae",
        "multiscale",
        "time_grad",
        "frequency_grad",
        "kl",
    }
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_default_vocoder_vae_has_requested_latent_map() -> None:
    config = VocoderVAEConfig()
    assert config.latent_shape == (16, 5, 16)
    assert config.latent_dim == 1_280
