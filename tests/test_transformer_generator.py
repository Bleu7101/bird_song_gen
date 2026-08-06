from __future__ import annotations

import torch

from bird_song.transformer.model import (
    ConditionalSpectrogramTransformer,
    TransformerGeneratorConfig,
    gaussian_patch_nll,
)


def small_config() -> TransformerGeneratorConfig:
    return TransformerGeneratorConfig(
        image_size=8,
        patch_size=4,
        channels=1,
        num_classes=3,
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )


def test_patchify_round_trip_uses_every_pixel() -> None:
    model = ConditionalSpectrogramTransformer(small_config())
    images = torch.linspace(-1, 1, steps=2 * 8 * 8).reshape(2, 1, 8, 8)
    patches = model.patchify(images)

    assert patches.shape == (2, 4, 16)
    assert torch.equal(model.unpatchify(patches), images)


def test_teacher_forced_distribution_matches_patch_targets() -> None:
    model = ConditionalSpectrogramTransformer(small_config())
    images = torch.rand(2, 1, 8, 8).mul(2).sub(1)
    labels = torch.tensor([0, 2])

    mean, log_scale = model(images, labels)
    loss = gaussian_patch_nll(images, mean, log_scale, model)

    assert mean.shape == (2, 4, 16)
    assert log_scale.shape == mean.shape
    assert torch.isfinite(loss)


def test_generation_returns_normalized_images() -> None:
    model = ConditionalSpectrogramTransformer(small_config())
    generated = model.generate(torch.tensor([0, 1, 2]), temperature=0.0)

    assert generated.shape == (3, 1, 8, 8)
    assert float(generated.amin()) >= -1.0
    assert float(generated.amax()) <= 1.0
