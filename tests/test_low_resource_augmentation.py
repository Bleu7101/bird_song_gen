import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from bird_song.augmentation.low_resource import (
    ExperimentCondition,
    ReplicateBlock,
    SharedSpectrogramMask,
    _bootstrap_mean_interval,
    _checkpoint_evidence_contract,
    _training_protocol,
    condition_run_dir,
    experiment_conditions,
    replicate_blocks,
    select_real_subset_rows,
)
from bird_song.augmentation.low_resource_report import (
    _subset_overlap_table,
    _validated_pool_provenance,
)
from bird_song.config import DEFAULT_CLASSES


def _manifest_rows(recordings_per_species: int = 5, clips_per_recording: int = 2) -> pd.DataFrame:
    rows = []
    for species_index, species in enumerate(DEFAULT_CLASSES):
        for recording_index in range(recordings_per_species):
            recording_id = f"{species_index}-{recording_index}"
            for clip_index in range(clips_per_recording):
                rows.append(
                    {
                        "split": "train",
                        "name": species,
                        "id": recording_id,
                        "relative_wav_path": f"wavfiles/{recording_id}-{clip_index}.wav",
                    }
                )
    return pd.DataFrame(rows)


def test_real_subset_is_balanced_deterministic_and_recording_distinct() -> None:
    rows = _manifest_rows()
    first = select_real_subset_rows(rows, real_per_species=3, seed=101)
    second = select_real_subset_rows(rows, real_per_species=3, seed=101)
    pd.testing.assert_frame_equal(first, second)
    assert first["name"].value_counts().to_dict() == {
        "American Robin": 3,
        "Northern Cardinal": 3,
        "Song Sparrow": 3,
    }
    assert first.groupby("name")["id"].nunique().eq(3).all()


def test_real_subset_rejects_more_rows_than_distinct_recordings() -> None:
    with pytest.raises(ValueError, match="distinct recording IDs"):
        select_real_subset_rows(_manifest_rows(recordings_per_species=2), real_per_species=3, seed=1)


def test_default_condition_matrix_has_one_control_and_six_synthetic_arms() -> None:
    conditions = experiment_conditions()
    assert len(conditions) == 7
    assert conditions[0] == ExperimentCondition("real_only", None, 0)
    assert {(condition.generator, condition.ratio_per_species) for condition in conditions[1:]} == {
        ("vae_v3", 50),
        ("vae_v3", 100),
        ("vae_v3", 200),
        ("diffusion", 50),
        ("diffusion", 100),
        ("diffusion", 200),
    }


def test_latin_square_pool_rotation_balances_all_pool_seeds() -> None:
    blocks = replicate_blocks()
    assert len(blocks) == 9
    assert {block.pool_seed for block in blocks} == {42, 123, 777}
    counts = pd.Series([block.pool_seed for block in blocks]).value_counts().to_dict()
    assert counts == {42: 3, 123: 3, 777: 3}
    for subset_seed in (101, 202, 303):
        assert {
            block.pool_seed for block in blocks if block.real_subset_seed == subset_seed
        } == {42, 123, 777}


def test_run_paths_keep_control_and_pool_specific_arms_separate(tmp_path: Path) -> None:
    block = ReplicateBlock(real_subset_seed=101, train_seed=42, pool_seed=123)
    control = condition_run_dir(tmp_path, block, ExperimentCondition("real_only", None, 0))
    synthetic = condition_run_dir(
        tmp_path,
        block,
        ExperimentCondition("vae_v3_plus_100", "vae_v3", 100),
    )
    assert control == tmp_path / "subset_101/train_42/real_only"
    assert synthetic == tmp_path / "subset_101/train_42/vae_v3/pool_123/ratio_100"


class _OneSpectrogram(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return torch.zeros((1, 20, 20), dtype=torch.float32), 0, "example.npy"


def test_shared_mask_is_reproducible_and_preserves_classifier_contract() -> None:
    dataset = SharedSpectrogramMask(_OneSpectrogram(), training=True)
    torch.manual_seed(123)
    first, label, path = dataset[0]
    torch.manual_seed(123)
    second, _, _ = dataset[0]
    assert torch.equal(first, second)
    assert first.shape == (1, 20, 20)
    assert set(torch.unique(first).tolist()).issubset({-1.0, 0.0})
    assert label == 0 and path == "example.npy"


def test_paired_bootstrap_interval_is_deterministic_and_contains_constant() -> None:
    assert _bootstrap_mean_interval([0.02] * 9) == pytest.approx((0.02, 0.02))
    first = _bootstrap_mean_interval([-0.01, 0.0, 0.02], seed=7, resamples=500)
    second = _bootstrap_mean_interval([-0.01, 0.0, 0.02], seed=7, resamples=500)
    assert first == second
    assert np.isfinite(first).all()


def test_subset_overlap_table_quantifies_shared_recordings() -> None:
    membership = pd.DataFrame(
        {
            "low_resource_subset_seed": [101, 101, 202, 202],
            "name": ["American Robin"] * 4,
            "id": ["a", "b", "b", "c"],
        }
    )
    overlap = _subset_overlap_table(membership)
    assert len(overlap) == 1
    row = overlap.iloc[0]
    assert row["species"] == "American Robin"
    assert row["subset_seed_a"] == 101
    assert row["subset_seed_b"] == 202
    assert row["shared_recording_ids"] == 1
    assert row["union_recording_ids"] == 3
    assert row["jaccard_overlap"] == pytest.approx(1 / 3)


def test_checkpoint_evidence_contract_rejects_protocol_drift(tmp_path: Path) -> None:
    block = ReplicateBlock(real_subset_seed=101, train_seed=42, pool_seed=42)
    identity = {"format_version": 1, "validation_counts": {"after": {"rows": 510}}}
    for condition in experiment_conditions((50,)):
        path = condition_run_dir(tmp_path, block, condition) / "best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        signature = {
            "condition": condition.condition,
            "generator": condition.generator,
            "ratio_per_species": condition.ratio_per_species,
            "real_subset_seed": block.real_subset_seed,
            "train_seed": block.train_seed,
            "pool_seed": block.pool_seed if condition.generator else None,
            "protocol_version": 4,
            "training_protocol": _training_protocol(),
            "validation_manifest": "manifests/generator_safe.csv",
            "validation_protocol": "manifests/generator_safe.protocol.json",
            "validation_protocol_identity": identity,
        }
        torch.save(
            {
                "condition": condition.condition,
                "generator": condition.generator,
                "ratio": condition.ratio_per_species,
                "real_subset_seed": block.real_subset_seed,
                "train_seed": block.train_seed,
                "pool_seed": block.pool_seed if condition.generator else None,
                "run_signature": signature,
            },
            path,
        )
    contract = _checkpoint_evidence_contract(
        run_root=tmp_path,
        ratios=(50,),
        real_subset_seeds=(101,),
        train_seeds=(42,),
        pool_seeds=(42,),
    )
    assert contract["protocol_version"] == 4
    assert contract["checkpoint_count"] == 3

    drift_path = condition_run_dir(
        tmp_path, block, ExperimentCondition("real_only", None, 0)
    ) / "best.pt"
    drifted = torch.load(drift_path, map_location="cpu", weights_only=True)
    drifted["run_signature"]["protocol_version"] = 3
    torch.save(drifted, drift_path)
    with pytest.raises(ValueError, match="run_signature.protocol_version"):
        _checkpoint_evidence_contract(
            run_root=tmp_path,
            ratios=(50,),
            real_subset_seeds=(101,),
            train_seeds=(42,),
            pool_seeds=(42,),
        )


def test_pool_provenance_is_quality_protocol_backed_and_rejects_action_drift() -> None:
    project_root = Path(__file__).resolve().parents[1]
    quality_path = (
        project_root / "reports/generator_checkpoint_evaluation_2026-08-12/protocol.json"
    )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    details = []
    action_fields = {
        "pool_reused_after_vae_bank_filter",
        "spectrograms_regenerated_after_vae_bank_filter",
    }
    for model in ("vae_v3", "diffusion"):
        contract = quality["generation_contracts"][model]
        for seed in (42, 123, 777):
            generation = {
                key: value for key, value in contract.items() if key not in action_fields
            }
            generation.update(
                {
                    "schema_version": 3,
                    "generator": model,
                    "seed": seed,
                    "samples_per_class": 200,
                }
            )
            details.append(
                {
                    "model": model,
                    "seed": seed,
                    "rows": 600,
                    "generation_identity": {"generation": generation},
                }
            )
    audit = {"pool_details": details}
    provenance = _validated_pool_provenance(
        input_audit=audit,
        generator_evaluation_protocol=quality,
        generator_evaluation_protocol_path=quality_path,
        project_root=project_root,
        pool_seeds=[42, 123, 777],
    )
    assert provenance["low_resource_pool_generation_performed"] is False
    vae = provenance["models"]["vae_v3"]
    assert vae["generation_contract"]["posterior_bank_counts"] == {
        "Northern Cardinal": 256,
        "Song Sparrow": 247,
        "American Robin": 256,
    }
    assert vae["generation_contract"]["vae_checkpoint_retrained"] is False
    assert vae["checkpoint_not_retrained"] is True
    assert vae["spectrograms_regenerated_after_vae_bank_filter"] is True
    assert provenance["models"]["diffusion"]["pool_reused_after_vae_bank_filter"] is True

    drifted = deepcopy(quality)
    drifted["generation_contracts"]["diffusion"][
        "spectrograms_regenerated_after_vae_bank_filter"
    ] = True
    with pytest.raises(ValueError, match="refresh contract mismatch"):
        _validated_pool_provenance(
            input_audit=audit,
            generator_evaluation_protocol=drifted,
            generator_evaluation_protocol_path=quality_path,
            project_root=project_root,
            pool_seeds=[42, 123, 777],
        )
