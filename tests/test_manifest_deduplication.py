from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_groups_are_transitive_across_id_and_audio_hash() -> None:
    module = _load_script("create_splits", "01_create_splits.py")
    rows = pd.DataFrame(
        [
            {"id": "same-id", "audio_sha256": "hash-one"},
            {"id": "same-id", "audio_sha256": "hash-two"},
            {"id": "other-id", "audio_sha256": "hash-two"},
            {"id": "independent", "audio_sha256": "hash-three"},
        ]
    )
    grouped = module.assign_split_groups(rows)
    assert grouped.loc[:2, "split_group"].nunique() == 1
    assert grouped.loc[3, "split_group"] != grouped.loc[0, "split_group"]


def test_content_dedup_preserves_test_then_validation_then_train(tmp_path: Path) -> None:
    module = _load_script("deduplicate_manifests", "01_deduplicate_manifests.py")
    dataset = tmp_path / "dataset"
    wavfiles = dataset / "wavfiles"
    wavfiles.mkdir(parents=True)
    for name, content in (("train.wav", b"same"), ("validation.wav", b"same"), ("test.wav", b"same"), ("unique.wav", b"unique")):
        (wavfiles / name).write_bytes(content)
    rows = pd.DataFrame(
        [
            {"split": split, "name": "A", "id": index, "filename": filename, "relative_wav_path": f"wavfiles/{filename}"}
            for index, (split, filename) in enumerate(
                (("train", "train.wav"), ("validation", "validation.wav"), ("test", "test.wav"), ("train", "unique.wav"))
            )
        ]
    )
    retained, removed, groups = module.deduplicate_manifest_rows(rows, dataset)
    assert groups == 1
    assert set(retained["filename"]) == {"test.wav", "unique.wav"}
    assert len(removed) == 2
    assert {record["retained_split"] for record in removed} == {"test"}
