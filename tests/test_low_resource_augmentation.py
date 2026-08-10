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
    condition_run_dir,
    experiment_conditions,
    replicate_blocks,
    select_real_subset_rows,
)
from bird_song.augmentation.low_resource_report import _subset_overlap_table
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
                        "audio_sha256": f"{species_index:02X}{recording_index:02X}{clip_index:02X}".ljust(64, "A"),
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
    assert not first["audio_sha256"].duplicated().any()


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
