from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bird_song.config import SpectrogramConfig
from bird_song.data import ManifestDataset
from bird_song.spectrogram_cache import audit_spectrogram_cache, canonicalize_spectrogram_cache


def _config() -> SpectrogramConfig:
    return SpectrogramConfig(n_mels=2, spectrogram_width=3)


def test_cache_audit_and_canonicalization_alias_identical_arrays(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    first = np.arange(6, dtype=np.float32).reshape(2, 3) / 10
    second = np.full((2, 3), -0.5, dtype=np.float32)
    np.save(root / "a.npy", first, allow_pickle=False)
    np.save(root / "b.npy", first, allow_pickle=False)
    np.save(root / "c.npy", second, allow_pickle=False)
    np.save(root / "orphan.npy", np.ones((2, 3), dtype=np.float32), allow_pickle=False)
    pd.DataFrame(
        [
            {"split": "train", "name": "A", "relative_wav_path": "one.wav", "relative_spectrogram_path": "a.npy"},
            {"split": "train", "name": "A", "relative_wav_path": "two.wav", "relative_spectrogram_path": "b.npy"},
            {"split": "test", "name": "A", "relative_wav_path": "three.wav", "relative_spectrogram_path": "c.npy"},
        ]
    ).to_csv(root / "spectrogram_manifest.csv", index=False)
    config_path = tmp_path / "spectrogram.json"
    config_path.write_text(json.dumps(_config().to_dict()), encoding="utf-8")

    _, before = audit_spectrogram_cache(root, _config())
    assert before["unique_object_count"] == 2
    assert before["redundant_physical_file_count"] == 1
    assert before["unreferenced_physical_file_count"] == 1

    after = canonicalize_spectrogram_cache(root, _config(), config_path, apply=True)
    rows = pd.read_csv(root / "spectrogram_manifest.csv")
    assert rows.loc[0, "relative_spectrogram_path"] == rows.loc[1, "relative_spectrogram_path"]
    assert len(list(root.rglob("*.npy"))) == 3
    assert (root / "orphan.npy").is_file()
    assert after["row_count"] == 3
    assert after["physical_path_count"] == 2
    assert after["unreferenced_physical_file_count"] == 1


def test_cached_manifest_dataset_supports_aliases_and_enforces_root_and_dtype(tmp_path: Path) -> None:
    config = _config()
    cache = tmp_path / "cache"
    objects = cache / "objects"
    objects.mkdir(parents=True)
    np.save(objects / "shared.npy", np.zeros((2, 3), dtype=np.float32), allow_pickle=False)
    pd.DataFrame(
        [
            {"split": "train", "relative_wav_path": "one.wav", "relative_spectrogram_path": "objects/shared.npy"},
            {"split": "train", "relative_wav_path": "two.wav", "relative_spectrogram_path": "objects/shared.npy"},
        ]
    ).to_csv(cache / "spectrogram_manifest.csv", index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"split": "train", "name": "A", "relative_wav_path": "one.wav"},
            {"split": "train", "name": "B", "relative_wav_path": "two.wav"},
        ]
    ).to_csv(manifest, index=False)

    dataset = ManifestDataset(manifest, None, ("A", "B"), config, spectrogram_cache_root=cache)
    assert dataset[0][2] == dataset[1][2]
    assert dataset[0][0].shape == (1, 2, 3)

    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((2, 3), dtype=np.float32), allow_pickle=False)
    escaped = pd.read_csv(cache / "spectrogram_manifest.csv")
    escaped.loc[0, "relative_spectrogram_path"] = "../outside.npy"
    escaped.to_csv(cache / "spectrogram_manifest.csv", index=False)
    with pytest.raises(ValueError, match="escapes"):
        ManifestDataset(manifest, None, ("A", "B"), config, spectrogram_cache_root=cache)

    escaped.loc[0, "relative_spectrogram_path"] = "objects/float64.npy"
    np.save(objects / "float64.npy", np.zeros((2, 3), dtype=np.float64), allow_pickle=False)
    escaped.to_csv(cache / "spectrogram_manifest.csv", index=False)
    strict = ManifestDataset(manifest, None, ("A", "B"), config, spectrogram_cache_root=cache)
    with pytest.raises(ValueError, match="float32"):
        strict[0]
