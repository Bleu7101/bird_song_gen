from __future__ import annotations

import json
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bird_song.classifier.model import build_classifier
from bird_song.generation.checkpoint_evaluation import (
    _frechet_distance,
    _manifold_metrics,
    _resampled_feature_metrics,
    audit_pools,
    evaluate,
)
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
from bird_song.generation.posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    POSTERIOR_BANK_EXPECTED_COUNTS,
    POSTERIOR_BANK_SOURCE_MANIFEST,
)


def test_checkpoint_architecture_parameter_counts_and_class_order() -> None:
    assert checkpoint_parameter_count(ConditionalVAE(3)) == VAE_PARAMETER_COUNT
    assert checkpoint_parameter_count(ConditionalUNet(dropout=0.1)) == DIFFUSION_PARAMETER_COUNT
    assert GENERATOR_CLASSES == ("Northern Cardinal", "Song Sparrow", "American Robin")


def test_crnn_forward_features_preserves_classifier_logits() -> None:
    inputs = torch.randn(2, 1, 128, 128)
    model = build_classifier("crnn", num_classes=3).eval()
    with torch.inference_mode():
        expected = model(inputs)
        actual = model.classifier(model.forward_features(inputs))
    assert torch.equal(expected, actual)


def test_evaluation_has_one_classifier_checkpoint() -> None:
    checkpoint_parameters = [
        name
        for name in signature(evaluate).parameters
        if name.endswith("_checkpoint")
        and name not in {"vae_checkpoint", "diffusion_checkpoint"}
    ]
    assert checkpoint_parameters == ["crnn_checkpoint"]


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


def test_pool_audit_uses_fresh_root_for_seed_42(tmp_path: Path) -> None:
    pool_root = tmp_path / "pools"
    classes = ("Northern Cardinal", "Song Sparrow", "American Robin")
    for model in ("vae_v3", "diffusion"):
        for seed in (42, 123, 777):
            root = pool_root / model / f"seed_{seed}"
            records = []
            for label_id, species in enumerate(classes):
                path = root / "classifier_input" / species.lower().replace(" ", "_") / "0000.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.full((1, 128, 128), (label_id + 1) / 10, dtype=np.float32))
                record = {
                    "species": species,
                    "relative_path": path.relative_to(root).as_posix(),
                    "generator": model,
                    "pool_rank": 0,
                    "sample_seed": deterministic_sample_seed(seed, label_id, 0),
                }
                if model == "vae_v3":
                    record.update(
                        {
                            "vae_anchor_index": label_id,
                            "vae_temperature": 0.35,
                            "vae_reparameterization": "mu + temperature * exp(0.5 * logvar) * epsilon",
                            "vae_posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                            "vae_bank_class_count": POSTERIOR_BANK_EXPECTED_COUNTS[species],
                            "vae_anchor_source_index": label_id,
                            "vae_anchor_relative_wav_path": f"wavfiles/{label_id}-{label_id}.wav",
                        }
                    )
                else:
                    record.update(
                        {
                            "sampler": "ddim",
                            "ddim_steps": 100,
                            "ddim_eta": 0.0,
                            "guidance_weight": 3.0,
                            "clamp_samples": 4.0,
                        }
                    )
                records.append(record)
            pd.DataFrame(records).to_csv(root / "manifest.csv", index=False)
            metadata = {
                "schema_version": 3,
                "generator": model,
                "classes": list(classes),
                "seed": seed,
                "samples_per_class": 1,
                "generation_batch_size": 8,
            }
            if model == "vae_v3":
                metadata.update(
                    {
                        "temperature": 0.35,
                        "reparameterization": "mu + temperature * exp(0.5 * logvar) * epsilon",
                        "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                        "posterior_bank_source_manifest": POSTERIOR_BANK_SOURCE_MANIFEST,
                        "posterior_bank_derivation": "filtered_existing_posterior_bank",
                        "posterior_bank_counts": POSTERIOR_BANK_EXPECTED_COUNTS,
                        "posterior_bank_inventory": {
                            inventory_species: {
                                "count": POSTERIOR_BANK_EXPECTED_COUNTS[inventory_species],
                                "source_indices": list(
                                    range(POSTERIOR_BANK_EXPECTED_COUNTS[inventory_species])
                                ),
                                "relative_wav_paths": [
                                    f"wavfiles/{inventory_label}-{index}.wav"
                                    for index in range(
                                        POSTERIOR_BANK_EXPECTED_COUNTS[inventory_species]
                                    )
                                ],
                            }
                            for inventory_label, inventory_species in enumerate(classes)
                        },
                        "vae_checkpoint_retrained": False,
                    }
                )
            else:
                metadata.update(
                    {
                        "sampler": "ddim",
                        "ddim_steps": 100,
                        "ddim_eta": 0.0,
                        "guidance_weight": 3.0,
                        "clamp_samples": 4.0,
                        "ema_state_dict": True,
                        "checkpoint_epoch": 34,
                        "checkpoint_best_validation_loss": 0.14470337276743556,
                        "checkpoint_selection": "validation_best",
                        "stored_sampler_overridden": "ddpm",
                    }
                )
            (root / "generation.json").write_text(json.dumps(metadata), encoding="utf-8")

    audit = audit_pools(tmp_path, pool_root, expected_samples=1)

    assert audit["seeds"] == [42, 123, 777]
    assert all(
        [pool["sample_count"] for pool in audit["models"][model]["pools"]] == [3, 3, 3]
        for model in ("vae_v3", "diffusion")
    )

    stale_manifest = pool_root / "vae_v3" / "seed_42" / "manifest.csv"
    stale_rows = pd.read_csv(stale_manifest)
    stale_rows.loc[0, "sample_seed"] = -1
    stale_rows.to_csv(stale_manifest, index=False)
    with pytest.raises(ValueError, match="non-canonical sample seeds"):
        audit_pools(tmp_path, pool_root, expected_samples=1)
    stale_rows.loc[0, "sample_seed"] = deterministic_sample_seed(42, 0, 0)
    stale_rows.to_csv(stale_manifest, index=False)

    stale_metadata = pool_root / "vae_v3" / "seed_42" / "generation.json"
    payload = json.loads(stale_metadata.read_text(encoding="utf-8"))
    payload.pop("posterior_bank_contract")
    stale_metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="VAE sampling contract mismatch"):
        audit_pools(tmp_path, pool_root, expected_samples=1)
