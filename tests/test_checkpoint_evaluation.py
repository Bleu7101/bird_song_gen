from __future__ import annotations

import numpy as np
import torch

from bird_song.classifier.model import build_classifier
from bird_song.generation.checkpoint_evaluation import _frechet_distance, _manifold_metrics, _resampled_feature_metrics
from bird_song.generation.checkpoint_models import (
    DIFFUSION_PARAMETER_COUNT,
    GENERATOR_CLASSES,
    VAE_PARAMETER_COUNT,
    ConditionalUNet,
    ConditionalVAE,
    checkpoint_parameter_count,
    classifier_scale_from_standardized,
)
from bird_song.generation.checkpoint_pool import deterministic_sample_seed


def test_checkpoint_architecture_parameter_counts_and_class_order() -> None:
    assert checkpoint_parameter_count(ConditionalVAE(3)) == VAE_PARAMETER_COUNT
    assert checkpoint_parameter_count(ConditionalUNet(dropout=0.1)) == DIFFUSION_PARAMETER_COUNT
    assert GENERATOR_CLASSES == ("Northern Cardinal", "Song Sparrow", "American Robin")


def test_forward_features_preserves_classifier_logits() -> None:
    inputs = torch.randn(2, 1, 128, 128)
    for architecture in ("crnn", "residual_cnn"):
        model = build_classifier(architecture, num_classes=3).eval()
        with torch.inference_mode():
            expected = model(inputs)
            actual = model.classifier(model.forward_features(inputs))
        assert torch.equal(expected, actual)


def test_scale_conversion_is_bounded_and_per_sample_max_is_zero_db() -> None:
    standardized = torch.tensor([[[[-1.0, 0.0], [1.0, 0.5]]]])
    converted = classifier_scale_from_standardized(standardized)
    assert converted.shape == standardized.shape
    assert float(converted.min()) >= -1.0
    assert float(converted.max()) <= 1.0
    assert torch.allclose(converted.amax(dim=(-2, -1)), torch.ones(1, 1))


def test_seed_derivation_and_metric_sanity() -> None:
    assert deterministic_sample_seed(123, 1, 4) == deterministic_sample_seed(123, 1, 4)
    real = np.random.default_rng(1).normal(size=(128, 64))
    generated = real.copy()
    assert _frechet_distance(real, generated) < 1e-8
    manifold = _manifold_metrics(real, generated, k=5)
    assert manifold["precision"] > 0.99
    assert manifold["recall"] > 0.99
    sampled = _resampled_feature_metrics(real, generated, seed=42, species_index=0, resamples=2, sample_size=16)
    assert sampled["resamples"] == 2
    assert sampled["sample_size"] == 16
    assert np.isfinite(sampled["frechet_mean"])
