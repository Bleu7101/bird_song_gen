from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from bird_song.generation import checkpoint_pool
from bird_song.generation.checkpoint_models import (
    VAE_REPARAMETERIZATION,
    VAE_TEMPERATURE,
    ConditionalVAE,
    diffusion_checkpoint_selection,
)
from bird_song.generation.posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    filter_existing_posterior_bank,
)


def test_vae_reparameterization_applies_recorded_temperature() -> None:
    mu = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    logvar = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    actual_generator = torch.Generator().manual_seed(123)
    expected_generator = torch.Generator().manual_seed(123)
    epsilon = torch.randn(mu.shape, generator=expected_generator, dtype=mu.dtype)

    actual = ConditionalVAE.reparameterize(mu, logvar, actual_generator)
    expected = mu + VAE_TEMPERATURE * torch.exp(0.5 * logvar) * epsilon

    assert torch.equal(actual, expected)
    assert torch.equal(
        ConditionalVAE.reparameterize(mu, logvar, torch.Generator().manual_seed(123), temperature=0.0),
        mu,
    )
    with pytest.raises(ValueError, match="temperature"):
        ConditionalVAE.reparameterize(mu, logvar, temperature=float("nan"))


def test_legacy_vae_record_is_not_reused_under_corrected_sampling_contract() -> None:
    seed = 42
    label_id = 1
    sample_index = 7
    base = {
        "generator": "vae_v3",
        "species": "Song Sparrow",
        "pool_rank": sample_index,
        "sample_seed": checkpoint_pool.deterministic_sample_seed(seed, label_id, sample_index),
        "vae_anchor_index": 11,
        "vae_temperature": VAE_TEMPERATURE,
    }

    assert not checkpoint_pool._record_matches_sampling_contract(
        base,
        "vae_v3",
        seed,
        label_id,
        sample_index,
    )
    corrected = {**base, "vae_reparameterization": VAE_REPARAMETERIZATION}
    assert not checkpoint_pool._record_matches_sampling_contract(
        corrected,
        "vae_v3",
        seed,
        label_id,
        sample_index,
    )
    filtered_bank_record = {
        **corrected,
        "vae_posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
        "vae_bank_class_count": 247,
        "vae_anchor_source_index": 12,
        "vae_anchor_relative_wav_path": "wavfiles/111645-3.wav",
    }
    assert checkpoint_pool._record_matches_sampling_contract(
        filtered_bank_record,
        "vae_v3",
        seed,
        label_id,
        sample_index,
    )


def test_vae_generation_metadata_records_corrected_formula(tmp_path: Path) -> None:
    counts = {0: 256, 1: 247, 2: 256}
    bank = {
        label_id: {
            "mu": torch.empty(count, 0),
            "source_indices": list(range(count)),
            "relative_wav_paths": [f"wavfiles/{label_id}-{index}.wav" for index in range(count)],
        }
        for label_id, count in counts.items()
    }
    checkpoint_pool._write_metadata(
        tmp_path,
        "vae_v3",
        seed=42,
        samples_per_species=200,
        checkpoint=Path("vae.pt"),
        posterior_bank=Path("posterior.pt"),
        generation_batch_size=8,
        bank=bank,
    )
    metadata_text = (tmp_path / "generation.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)

    assert metadata["schema_version"] == 3
    assert metadata["temperature"] == VAE_TEMPERATURE
    assert metadata["reparameterization"] == VAE_REPARAMETERIZATION
    assert metadata["generation_batch_size"] == 8
    assert metadata["posterior_bank_contract"] == POSTERIOR_BANK_CONTRACT
    assert metadata["posterior_bank_counts"] == {
        "Northern Cardinal": 256,
        "Song Sparrow": 247,
        "American Robin": 256,
    }
    assert metadata["posterior_bank_inventory"]["Song Sparrow"]["count"] == 247
    assert metadata["posterior_bank_inventory"]["Song Sparrow"]["source_indices"][12] == 12
    assert metadata["posterior_bank_inventory"]["Song Sparrow"]["relative_wav_paths"][12].startswith(
        "wavfiles/"
    )
    assert metadata["vae_checkpoint_retrained"] is False


def test_existing_posterior_bank_filter_preserves_only_unique_train_anchors() -> None:
    label_to_id = {"Northern Cardinal": 0, "Song Sparrow": 1, "American Robin": 2}
    banks = {}
    manifest_rows = []
    for species, label_id in label_to_id.items():
        slug = species.lower().replace(" ", "_")
        paths = [f"C:\\old\\train\\{slug}\\{label_id}00.npy", f"C:\\old\\train\\{slug}\\{label_id}01.npy"]
        mu = torch.arange(2 * 16 * 16 * 16, dtype=torch.float32).reshape(2, 16, 16, 16)
        logvar = mu + 100.0
        banks[label_id] = {"mu": mu, "logvar": logvar, "paths": paths}
        manifest_rows.append(
            {
                "split": "train",
                "name": species,
                "filename": f"{label_id}00.wav",
                "relative_wav_path": f"wavfiles/{label_id}00.wav",
            }
        )
        if species != "Song Sparrow":
            manifest_rows.append(
                {
                    "split": "train",
                    "name": species,
                    "filename": f"{label_id}01.wav",
                    "relative_wav_path": f"wavfiles/{label_id}01.wav",
                }
            )
    package = {
        "banks": banks,
        "temperature": VAE_TEMPERATURE,
        "label_to_id": label_to_id,
        "fitted_split": "train",
        "sampling_type": "per_species_posterior_anchor_mixture",
    }

    filtered = filter_existing_posterior_bank(
        package,
        pd.DataFrame(manifest_rows),
        enforce_canonical_counts=False,
    )

    assert filtered["posterior_bank_contract"] == POSTERIOR_BANK_CONTRACT
    assert filtered["vae_checkpoint_retrained"] is False
    assert filtered["counts"] == {
        "Northern Cardinal": 2,
        "Song Sparrow": 1,
        "American Robin": 2,
    }
    song = filtered["banks"][1]
    assert torch.equal(song["mu"], banks[1]["mu"][:1])
    assert torch.equal(song["logvar"], banks[1]["logvar"][:1])
    assert song["source_indices"] == [0]
    assert song["relative_wav_paths"] == ["wavfiles/100.wav"]
    assert filtered["removed_anchors"]["Song Sparrow"] == [
        {"source_index": 1, "source_anchor_name": "101"}
    ]


def test_diffusion_checkpoint_selection_requires_the_recorded_best_epoch() -> None:
    checkpoint = {
        "epoch": 34,
        "best_val_loss": 0.14470337276743556,
        "history": {
            "epoch": [33, 34],
            "val_loss": [0.15913209835917963, 0.14470337276743556],
        },
    }
    assert diffusion_checkpoint_selection(checkpoint) == {
        "checkpoint_epoch": 34,
        "checkpoint_best_validation_loss": 0.14470337276743556,
        "checkpoint_selection": "validation_best",
    }
    checkpoint["epoch"] = 33
    with pytest.raises(ValueError, match="canonical validation-best"):
        diffusion_checkpoint_selection(checkpoint)


def test_benchmark_uses_fresh_in_memory_samples_and_records_repeat_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = torch.nn.Linear(1, 1)
    sample_calls: list[tuple[int, tuple[int, ...]]] = []
    synchronize_calls: list[str] = []
    clock = iter((10.0, 12.0, 20.0, 22.0))

    monkeypatch.setattr(checkpoint_pool.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(checkpoint_pool.torch.cuda, "memory_allocated", lambda _device: 100)
    monkeypatch.setattr(checkpoint_pool.torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(checkpoint_pool.torch.cuda, "max_memory_allocated", lambda _device: 150)
    fake_bank = {
        label_id: {
            "mu": torch.zeros(1, 4),
            "source_indices": [label_id],
            "relative_wav_paths": [f"wavfiles/{label_id}.wav"],
        }
        for label_id in range(len(checkpoint_pool.GENERATOR_CLASSES))
    }
    monkeypatch.setattr(
        checkpoint_pool,
        "_prepare_generator",
        lambda *_args, **_kwargs: (fake_model, fake_bank, None, {}),
    )
    monkeypatch.setattr(
        checkpoint_pool,
        "_sample_standardized_chunk",
        lambda _model, _generator_model, _bank, _schedule, _seed, label_id, indices, _device: (
            sample_calls.append((label_id, tuple(indices))) or torch.empty(0),
            [],
        ),
    )
    monkeypatch.setattr(
        checkpoint_pool,
        "_synchronize_cuda",
        lambda _device: synchronize_calls.append("sync"),
    )
    monkeypatch.setattr(checkpoint_pool.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        checkpoint_pool,
        "_hardware_metadata",
        lambda _device: {"device": "cuda", "cuda_device_name": "test GPU"},
    )

    output = tmp_path / "benchmark.json"
    result = checkpoint_pool.benchmark_generation(
        model="vae_v3",
        checkpoint=tmp_path / "vae.pt",
        seed=42,
        samples_per_species=2,
        device=torch.device("cuda"),
        posterior_bank=tmp_path / "posterior.pt",
        batch_size=2,
        warmup_batches=1,
        repeats=2,
        metadata_output=output,
    )

    assert result["existing_arrays_used"] is False
    assert result["fresh_in_memory_generation_each_repeat"] is True
    assert result["samples_per_repeat"] == 6
    assert result["repeat_count"] == 2
    assert result["repeat_results"] == [
        {
            "repeat": 1,
            "sample_count": 6,
            "seconds": 2.0,
            "samples_per_second": 3.0,
            "peak_cuda_memory_bytes": 150,
            "incremental_peak_cuda_memory_bytes": 50,
        },
        {
            "repeat": 2,
            "sample_count": 6,
            "seconds": 2.0,
            "samples_per_second": 3.0,
            "peak_cuda_memory_bytes": 150,
            "incremental_peak_cuda_memory_bytes": 50,
        },
    ]
    assert result["precision"] == "float32"
    assert result["batch_size"] == 2
    assert result["warmup_batches"] == 1
    assert result["cuda_synchronized"] is True
    assert result["settings"]["temperature"] == VAE_TEMPERATURE
    assert result["settings"]["reparameterization"] == VAE_REPARAMETERIZATION
    assert len(sample_calls) == 7  # one warm-up plus three species in each of two repeats
    assert len(synchronize_calls) == 5  # after warm-up and around both timed repeats
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not list(tmp_path.rglob("*.npy"))
