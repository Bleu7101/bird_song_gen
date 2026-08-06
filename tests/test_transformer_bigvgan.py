from __future__ import annotations

import torch

from bird_song.transformer.model import (
    ConditionalSpectrogramTransformer,
    TransformerGeneratorConfig,
    gaussian_patch_nll,
)


def test_production_transformer_configs_use_rectangular_patch_grids() -> None:
    wide = TransformerGeneratorConfig()
    fine_frequency = TransformerGeneratorConfig(patch_height=8, patch_width=16)
    assert (wide.height, wide.width) == (80, 256)
    assert wide.patch_count == 80
    assert fine_frequency.patch_count == 160


def test_rectangular_patchify_round_trip() -> None:
    config = TransformerGeneratorConfig(
        height=8,
        width=16,
        patch_height=4,
        patch_width=4,
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )
    model = ConditionalSpectrogramTransformer(config)
    images = torch.linspace(-1, 1, steps=2 * 8 * 16).reshape(2, 1, 8, 16)
    patches = model.patchify(images)
    assert patches.shape == (2, 8, 16)
    assert torch.equal(model.unpatchify(patches), images)


def test_rectangular_teacher_forcing_and_generation() -> None:
    config = TransformerGeneratorConfig(
        height=8,
        width=16,
        patch_height=4,
        patch_width=4,
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )
    model = ConditionalSpectrogramTransformer(config)
    images = torch.rand(2, 1, 8, 16).mul(2).sub(1)
    labels = torch.tensor([0, 2])
    mean, log_scale = model(images, labels)
    loss = gaussian_patch_nll(images, mean, log_scale, model)
    generated = model.generate(labels, temperature=0.0)
    assert mean.shape == (2, config.patch_count, config.patch_dimension)
    assert log_scale.shape == mean.shape
    assert torch.isfinite(loss)
    assert generated.shape == images.shape
    assert float(generated.amin()) >= -1.0
    assert float(generated.amax()) <= 1.0
