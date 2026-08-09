from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bird_song.augmentation.data import GeneratedSpectrogramDataset, audit_generated_pool
from bird_song.augmentation.experiment import (
    _guard_legacy_overwrite,
    _common_run_signature,
    _validate_baseline_test_manifest,
    _validate_existing_run,
    select_ratio,
)
from bird_song.config import SpectrogramConfig
from bird_song.spectrogram_cache import array_sha256, sha256_file


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
                    "array_sha256": array_sha256(array),
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
    assert audit["rows"] == audit["unique_arrays"] == 4
    assert audit["rows_per_species"] == {"A": 2, "B": 2}


def test_generated_pool_audit_rejects_hash_mismatch(tmp_path: Path) -> None:
    root, manifest, config = _pool(tmp_path)
    rows = pd.read_csv(manifest)
    rows.loc[0, "array_sha256"] = "0" * 64
    rows.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="hash does not match"):
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


def test_legacy_existing_run_is_not_silently_reused(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    torch.save({"generator": "vae_v3"}, output / "best.pt")
    (output / "config.json").write_text(json.dumps({"real_rows": 2339}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible"):
        _validate_existing_run(output, {"protocol_version": 2})
    with pytest.raises(RuntimeError, match="Refusing to overwrite legacy evidence"):
        _guard_legacy_overwrite(output / "best.pt")


def test_baseline_delta_requires_the_same_logical_test_clips(tmp_path: Path) -> None:
    project = tmp_path / "project"
    manifests = project / "manifests"
    manifests.mkdir(parents=True)
    baseline_manifest = manifests / "baseline.csv"
    pd.DataFrame(
        [{"split": "test", "name": "A", "relative_wav_path": "wavfiles/a.wav"}]
    ).to_csv(baseline_manifest, index=False)
    baseline_payload = {
        "manifest": {
            "path": "manifests/baseline.csv",
            "sha256": sha256_file(baseline_manifest),
            "sample_count": 1,
        }
    }
    equivalent = manifests / "equivalent.csv"
    pd.DataFrame(
        [
            {
                "split": "test",
                "name": "A",
                "relative_wav_path": "wavfiles/a.wav",
                "audio_sha256": "A" * 64,
            }
        ]
    ).to_csv(equivalent, index=False)
    assert _validate_baseline_test_manifest(baseline_payload, equivalent, project) == baseline_payload["manifest"]

    different = manifests / "different.csv"
    pd.DataFrame(
        [{"split": "test", "name": "A", "relative_wav_path": "wavfiles/b.wav"}]
    ).to_csv(different, index=False)
    with pytest.raises(ValueError, match="different logical clips"):
        _validate_baseline_test_manifest(baseline_payload, different, project)


def test_common_run_signature_keeps_protocol_and_drops_only_arm_identity() -> None:
    first = {
        "generator": "vae_v3",
        "ratio_per_species": 50,
        "seed": 42,
        "pool_manifest_sha256": "A" * 64,
        "steps": 1440,
        "training_protocol": {"optimizer": "AdamW"},
    }
    second = {
        **first,
        "generator": "diffusion",
        "ratio_per_species": 200,
        "seed": 777,
        "pool_manifest_sha256": "B" * 64,
    }
    assert _common_run_signature(first) == _common_run_signature(second)
    second["steps"] = 720
    assert _common_run_signature(first) != _common_run_signature(second)
