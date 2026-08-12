from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bird_song.augmentation.data import GeneratedSpectrogramDataset, audit_generated_pool
from bird_song.augmentation.experiment import select_ratio
from bird_song.augmentation.low_resource import (
    ExperimentCondition,
    PoolReference,
    ReplicateBlock,
    _pool_generation_identity,
    _run_signature,
    _validate_existing_run,
    pool_reference,
)
from bird_song.config import SpectrogramConfig
from bird_song.generation.checkpoint_pool import deterministic_sample_seed
from bird_song.generation.posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    POSTERIOR_BANK_EXPECTED_COUNTS,
    POSTERIOR_BANK_SOURCE_MANIFEST,
)
from bird_song.generation.checkpoint_models import (
    DIFFUSION_CLAMP,
    DIFFUSION_DDIM_ETA,
    DIFFUSION_DDIM_STEPS,
    DIFFUSION_GUIDANCE,
    GENERATOR_CLASSES,
    VAE_REPARAMETERIZATION,
    VAE_TEMPERATURE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pool(tmp_path: Path) -> tuple[Path, Path, SpectrogramConfig]:
    root = tmp_path / "generated"
    config = SpectrogramConfig(n_mels=2, spectrogram_width=3)
    rows = []
    for species_index, species in enumerate(("A", "B")):
        for rank in range(2):
            relative = f"model\\classifier_input\\{species.lower()}\\{rank:04d}.npy"
            path = root / Path(relative.replace("\\", "/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.full((2, 3), -0.8 + 0.2 * (species_index * 2 + rank), dtype=np.float32)
            np.save(path, array, allow_pickle=False)
            rows.append(
                {
                    "species": species,
                    "relative_path": relative,
                    "pool_rank": rank,
                    "generator": "vae_v3",
                    "vae_temperature": 0.35,
                    "vae_reparameterization": "mu_plus_temperature_std_epsilon",
                }
            )
    manifest = root / "model" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return root, manifest, config


def test_generated_dataset_remaps_labels_and_selects_rank_prefix(tmp_path: Path) -> None:
    root, manifest, config = _pool(tmp_path)
    dataset = GeneratedSpectrogramDataset(manifest, root, ("B", "A"), config, 1)
    assert len(dataset) == 2
    assert [dataset[index][1] for index in range(2)] == [1, 0]
    assert all("\\" not in path.as_posix() for path in dataset.paths)
    assert dataset[0][0].shape == (1, 2, 3)
    audit = audit_generated_pool(manifest, root, ("A", "B"), config)
    assert audit["rows"] == audit["validated_arrays"] == 4
    assert audit["rows_per_species"] == {"A": 2, "B": 2}


def test_generated_pool_audit_rejects_missing_array(tmp_path: Path) -> None:
    root, manifest, config = _pool(tmp_path)
    rows = pd.read_csv(manifest)
    (root / Path(str(rows.loc[0, "relative_path"]).replace("\\", "/"))).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        audit_generated_pool(manifest, root, ("A", "B"), config)


def test_generated_dataset_rejects_unknown_species(tmp_path: Path) -> None:
    root, manifest, config = _pool(tmp_path)
    rows = pd.read_csv(manifest)
    rows.loc[0, "species"] = "Unknown"
    rows.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="unknown species"):
        GeneratedSpectrogramDataset(manifest, root, ("A", "B"), config, 1)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((3, 3), dtype=np.float32), "must be"),
        (np.full((2, 3), 2.0, dtype=np.float32), "outside"),
    ],
)
def test_generated_dataset_rejects_bad_shape_or_range(
    tmp_path: Path, array: np.ndarray, message: str
) -> None:
    root, manifest, config = _pool(tmp_path)
    rows = pd.read_csv(manifest)
    target = root / Path(str(rows.loc[0, "relative_path"]).replace("\\", "/"))
    np.save(target, array, allow_pickle=False)
    dataset = GeneratedSpectrogramDataset(manifest, root, ("A", "B"), config, 1)
    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_ratio_selection_prefers_smallest_ratio_across_full_tie_band() -> None:
    assert select_ratio({50: 0.8985, 100: 0.9000, 200: 0.8990}) == 50
    assert select_ratio({50: 0.89, 100: 0.90, 200: 0.91}) == 200
    with pytest.raises(ValueError):
        select_ratio({})


def test_all_pool_seeds_use_the_fresh_generation_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    pool_root = tmp_path / "pools"
    for seed in (42, 123, 777):
        reference = pool_reference(project_root, pool_root, "vae_v3", seed)
        expected = pool_root / "vae_v3" / f"seed_{seed}"
        assert reference.cache_root == expected
        assert reference.manifest == expected / "manifest.csv"
        assert reference.generation_metadata == expected / "generation.json"


def _canonical_pool_reference(
    tmp_path: Path,
    model: str = "vae_v3",
) -> tuple[PoolReference, SpectrogramConfig]:
    root = tmp_path / model
    root.mkdir(parents=True)
    config = SpectrogramConfig(n_mels=2, spectrogram_width=3)
    rows = []
    for label_id, species in enumerate(GENERATOR_CLASSES):
        for rank in range(200):
            row = {
                "species": species,
                "relative_path": f"classifier_input/{species.lower().replace(' ', '_')}/{rank:04d}.npy",
                "pool_rank": rank,
                "sample_seed": deterministic_sample_seed(42, label_id, rank),
                "generator": model,
            }
            if model == "vae_v3":
                row.update(
                    {
                        "vae_temperature": VAE_TEMPERATURE,
                        "vae_reparameterization": VAE_REPARAMETERIZATION,
                        "vae_posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                        "vae_bank_class_count": POSTERIOR_BANK_EXPECTED_COUNTS[species],
                        "vae_anchor_source_index": rank % POSTERIOR_BANK_EXPECTED_COUNTS[species],
                        "vae_anchor_relative_wav_path": f"wavfiles/{label_id}-{rank}.wav",
                    }
                )
            else:
                row.update(
                    {
                        "sampler": "ddim",
                        "ddim_steps": DIFFUSION_DDIM_STEPS,
                        "ddim_eta": DIFFUSION_DDIM_ETA,
                        "guidance_weight": DIFFUSION_GUIDANCE,
                        "clamp_samples": DIFFUSION_CLAMP,
                    }
                )
            rows.append(row)
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    payload = {
        "schema_version": 3,
        "generator": model,
        "classes": list(GENERATOR_CLASSES),
        "seed": 42,
        "samples_per_class": 200,
        "generation_batch_size": 8,
    }
    if model == "vae_v3":
        payload.update(
            {
                "temperature": VAE_TEMPERATURE,
                "reparameterization": VAE_REPARAMETERIZATION,
                "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                "posterior_bank_source_manifest": POSTERIOR_BANK_SOURCE_MANIFEST,
                "posterior_bank_counts": POSTERIOR_BANK_EXPECTED_COUNTS,
                "vae_checkpoint_retrained": False,
            }
        )
    else:
        payload.update(
            {
                "sampler": "ddim",
                "ddim_steps": DIFFUSION_DDIM_STEPS,
                "ddim_eta": DIFFUSION_DDIM_ETA,
                "guidance_weight": DIFFUSION_GUIDANCE,
                "clamp_samples": DIFFUSION_CLAMP,
                "ema_state_dict": True,
                "checkpoint_epoch": 34,
                "checkpoint_best_validation_loss": 0.14470337276743556,
                "checkpoint_selection": "validation_best",
                "stored_sampler_overridden": "ddpm",
            }
        )
    metadata = root / "generation.json"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    return (
        PoolReference(
            model=model,
            seed=42,
            manifest=manifest,
            cache_root=root,
            generation_metadata=metadata,
        ),
        config,
    )


@pytest.mark.parametrize("model", ["vae_v3", "diffusion"])
def test_pool_identity_accepts_only_the_corrected_canonical_contract(
    tmp_path: Path,
    model: str,
) -> None:
    reference, _ = _canonical_pool_reference(tmp_path, model)
    identity = _pool_generation_identity(reference)
    assert identity["generation"]["schema_version"] == 3
    assert identity["generation"]["samples_per_class"] == 200
    assert identity["manifest_config"]["rows_per_species"] == {
        species: 200 for species in sorted(GENERATOR_CLASSES)
    }


@pytest.mark.parametrize(
    ("model", "field", "invalid"),
    [
        ("vae_v3", "schema_version", 2),
        ("vae_v3", "generator", "diffusion"),
        ("vae_v3", "seed", 123),
        ("vae_v3", "classes", list(reversed(GENERATOR_CLASSES))),
        ("vae_v3", "samples_per_class", 199),
        ("vae_v3", "temperature", 1.0),
        ("vae_v3", "reparameterization", "mu + exp(0.5 * logvar) * epsilon"),
        ("vae_v3", "posterior_bank_contract", "legacy_bank"),
        ("vae_v3", "posterior_bank_source_manifest", "manifests/legacy.csv"),
        ("vae_v3", "posterior_bank_counts", {}),
        ("vae_v3", "vae_checkpoint_retrained", True),
        ("diffusion", "sampler", "ddpm"),
        ("diffusion", "ddim_steps", 50),
        ("diffusion", "ddim_eta", 1.0),
        ("diffusion", "guidance_weight", 1.0),
        ("diffusion", "clamp_samples", 1.0),
        ("diffusion", "ema_state_dict", False),
    ],
)
def test_pool_identity_rejects_noncanonical_generation_metadata(
    tmp_path: Path,
    model: str,
    field: str,
    invalid: object,
) -> None:
    reference, _ = _canonical_pool_reference(tmp_path, model)
    payload = json.loads(reference.generation_metadata.read_text(encoding="utf-8"))
    payload[field] = invalid
    reference.generation_metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        _pool_generation_identity(reference)


@pytest.mark.parametrize(
    ("model", "field", "invalid"),
    [
        ("vae_v3", "generator", "diffusion"),
        ("vae_v3", "sample_seed", 1),
        ("vae_v3", "vae_temperature", 1.0),
        ("vae_v3", "vae_reparameterization", "mu + std * epsilon"),
        ("vae_v3", "vae_posterior_bank_contract", "legacy_bank"),
        ("vae_v3", "vae_bank_class_count", 2560),
        ("vae_v3", "vae_anchor_source_index", -1),
        ("vae_v3", "vae_anchor_relative_wav_path", "C:/old/anchor.wav"),
        ("diffusion", "sampler", "ddpm"),
        ("diffusion", "ddim_steps", 50),
        ("diffusion", "ddim_eta", 1.0),
        ("diffusion", "guidance_weight", 1.0),
        ("diffusion", "clamp_samples", 1.0),
    ],
)
def test_pool_identity_rejects_noncanonical_manifest_settings(
    tmp_path: Path,
    model: str,
    field: str,
    invalid: object,
) -> None:
    reference, _ = _canonical_pool_reference(tmp_path, model)
    rows = pd.read_csv(reference.manifest)
    rows.loc[0, field] = invalid
    rows.to_csv(reference.manifest, index=False)
    with pytest.raises(ValueError, match="non-canonical"):
        _pool_generation_identity(reference)


def test_pool_identity_rejects_manifest_count_drift(tmp_path: Path) -> None:
    reference, _ = _canonical_pool_reference(tmp_path)
    rows = pd.read_csv(reference.manifest).iloc[:-1]
    rows.to_csv(reference.manifest, index=False)
    with pytest.raises(ValueError, match="exactly 200 rows"):
        _pool_generation_identity(reference)


def test_low_resource_run_signature_rejects_generation_contract_drift(tmp_path: Path) -> None:
    reference, config = _canonical_pool_reference(tmp_path)
    project_root = PROJECT_ROOT
    common = {
        "project_root": project_root,
        "block": ReplicateBlock(real_subset_seed=101, train_seed=42, pool_seed=42),
        "condition": ExperimentCondition("vae_v3_plus_50", "vae_v3", 50),
        "source_train_manifest": tmp_path / "train.csv",
        "subset_manifest": tmp_path / "subset.csv",
        "validation_manifest": (
            PROJECT_ROOT
            / "manifests/content_safe_v2/full_dataset_validation_generator_safe.csv"
        ),
        "cache_root": tmp_path / "cache",
        "spectrogram_config_path": tmp_path / "spectrogram.json",
        "config": config,
        "pool": reference,
        "steps": 1_440,
        "validate_every": 36,
        "batch_size": 64,
        "workers": 0,
        "real_per_species": 50,
        "device": torch.device("cpu"),
    }
    original = _run_signature(**common)
    assert original["protocol_version"] == 4
    assert original["validation_protocol_identity"]["validation_counts"]["after"]["rows"] == 510
    assert original["pool_generation_identity"]["generation"]["temperature"] == 0.35

    metadata = json.loads(reference.generation_metadata.read_text(encoding="utf-8"))
    metadata["temperature"] = 0.20
    reference.generation_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        _run_signature(**common)

    metadata["temperature"] = VAE_TEMPERATURE
    reference.generation_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    rows = pd.read_csv(reference.manifest)
    rows["vae_temperature"] = 0.20
    rows.to_csv(reference.manifest, index=False)
    with pytest.raises(ValueError, match="non-canonical"):
        _run_signature(**common)


def test_protocol_three_run_cannot_be_reused_by_protocol_four(tmp_path: Path) -> None:
    output = tmp_path / "old_run"
    output.mkdir()
    old_signature = {"protocol_version": 3, "validation_manifest": "legacy.csv"}
    torch.save({"run_signature": old_signature}, output / "best.pt")
    (output / "config.json").write_text(
        json.dumps({"run_signature": old_signature}),
        encoding="utf-8",
    )
    expected = {
        "protocol_version": 4,
        "validation_manifest": "full_dataset_validation_generator_safe.csv",
        "validation_protocol_identity": {"format_version": 1},
    }
    with pytest.raises(RuntimeError, match="incompatible"):
        _validate_existing_run(output, expected)
