from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bird_song.generator_safe_validation import (
    EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS,
    EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS,
    EXPECTED_TEST_ROWS,
    load_generator_safe_validation_identity,
    prepare_generator_safe_validation,
    protocol_path_for_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VALIDATION = (
    PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_validation.csv"
)
SOURCE_TEST = PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_test.csv"
HISTORICAL_MANIFEST = PROJECT_ROOT / "manifests/full_dataset_manifest.csv"
EXACT_DUPLICATE_PROTOCOL = PROJECT_ROOT / "manifests/content_safe_v2/protocol.json"


def _prepare(tmp_path: Path) -> tuple[Path, Path, dict]:
    output = tmp_path / "full_dataset_validation_generator_safe.csv"
    protocol = prepare_generator_safe_validation(
        project_root=PROJECT_ROOT,
        source_validation_manifest=SOURCE_VALIDATION,
        source_test_manifest=SOURCE_TEST,
        historical_manifest=HISTORICAL_MANIFEST,
        exact_duplicate_protocol=EXACT_DUPLICATE_PROTOCOL,
        output_validation_manifest=output,
    )
    return output, protocol_path_for_manifest(output), protocol


def test_preparation_is_deterministic_and_validation_only(tmp_path: Path) -> None:
    source_validation_before = pd.read_csv(SOURCE_VALIDATION)
    source_test_before = pd.read_csv(SOURCE_TEST)
    output, protocol_path, protocol = _prepare(tmp_path)
    first_csv = output.read_text(encoding="utf-8")
    first_protocol = protocol_path.read_text(encoding="utf-8")

    _, _, repeated = _prepare(tmp_path)

    assert output.read_text(encoding="utf-8") == first_csv
    assert protocol_path.read_text(encoding="utf-8") == first_protocol
    assert repeated == protocol
    assert pd.read_csv(SOURCE_VALIDATION).equals(source_validation_before)
    assert pd.read_csv(SOURCE_TEST).equals(source_test_before)
    assert protocol["validation_counts"]["before"]["rows"] == 519
    assert protocol["validation_counts"]["after"] == {
        "rows": EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS,
        "species": {
            "American Robin": 163,
            "Northern Cardinal": 172,
            "Song Sparrow": 175,
        },
    }
    assert protocol["excluded_validation_row_count"] == 9
    assert (
        protocol["historical_train_counterpart_row_count"]
        == EXPECTED_HISTORICAL_TRAIN_COUNTERPART_ROWS
    )
    assert protocol["test_unchanged"]["rows"] == EXPECTED_TEST_ROWS
    assert protocol["test_unchanged"]["excluded_rows"] == 0
    assert len(pd.read_csv(output)) == EXPECTED_GENERATOR_SAFE_VALIDATION_ROWS


def test_semantic_identity_rejects_manifest_drift(tmp_path: Path) -> None:
    output, _, _ = _prepare(tmp_path)
    identity = load_generator_safe_validation_identity(output, PROJECT_ROOT)
    assert identity["validation_counts"]["after"]["rows"] == 510
    assert len(identity["excluded_validation_relative_wav_paths"]) == 9
    assert len(identity["historical_train_counterparts"]) == 17

    rows = pd.read_csv(output).iloc[:-1]
    rows.to_csv(output, index=False)
    with pytest.raises(ValueError, match="wrong row count"):
        load_generator_safe_validation_identity(output, PROJECT_ROOT)


def test_semantic_identity_rejects_protocol_drift(tmp_path: Path) -> None:
    output, protocol_path, protocol = _prepare(tmp_path)
    protocol["test_unchanged"]["excluded_rows"] = 1
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="preserve the held-out test set"):
        load_generator_safe_validation_identity(output, PROJECT_ROOT)
