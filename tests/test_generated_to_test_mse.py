from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bird_song.evaluation.generated_to_test_mse import evaluate_generated_to_test


def _write_array(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((1, 128, 128), value, dtype=np.float32))


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    generated_root = tmp_path / "generated"
    test_root = tmp_path / "test"
    _write_array(generated_root / "A" / "g0.npy", 0.25)
    _write_array(generated_root / "B" / "g0.npy", -0.25)
    _write_array(test_root / "a0.npy", 0.0)
    _write_array(test_root / "a1.npy", 0.5)
    _write_array(test_root / "b0.npy", -0.5)
    generated_manifest = tmp_path / "generated.csv"
    generated_manifest.write_text(
        "species,relative_path\nA,A/g0.npy\nB,B/g0.npy\n", encoding="utf-8"
    )
    test_manifest = tmp_path / "test.csv"
    test_manifest.write_text(
        "species,relative_path\nA,a0.npy\nA,a1.npy\nB,b0.npy\n", encoding="utf-8"
    )
    return generated_manifest, generated_root, test_manifest, test_root


def test_evaluation_uses_same_species_and_reports_both_protocols(tmp_path: Path) -> None:
    generated_manifest, generated_root, test_manifest, test_root = _fixture_inputs(tmp_path)

    result = evaluate_generated_to_test(
        generated_manifest=generated_manifest,
        generated_root=generated_root,
        test_manifest=test_manifest,
        test_root=test_root,
    )

    assert result.per_sample["generated_to_test"]["mse"].tolist() == pytest.approx([0.0625, 0.0625])
    assert result.per_sample["real_to_real"]["mse"].tolist() == pytest.approx([0.25, 0.25, np.nan], nan_ok=True)
    summary = result.summary.set_index(["protocol", "species"])
    assert summary.loc[("generated_to_test", "A"), "count"] == 1
    assert summary.loc[("real_to_real", "B"), "count"] == 0
    assert summary.loc[("generated_to_test", "overall"), "median"] == pytest.approx(0.0625)
    assert {"mean", "std", "median", "q1", "q3", "min", "max"}.issubset(result.summary.columns)


def test_evaluation_rejects_wrong_shape_or_domain(tmp_path: Path) -> None:
    generated_manifest, generated_root, test_manifest, test_root = _fixture_inputs(tmp_path)
    np.save(generated_root / "A" / "g0.npy", np.zeros((128, 128), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        evaluate_generated_to_test(
            generated_manifest=generated_manifest,
            generated_root=generated_root,
            test_manifest=test_manifest,
            test_root=test_root,
        )


def test_report_writer_persists_protocol_and_provenance(tmp_path: Path) -> None:
    generated_manifest, generated_root, test_manifest, test_root = _fixture_inputs(tmp_path)
    output = tmp_path / "report"
    result = evaluate_generated_to_test(
        generated_manifest=generated_manifest,
        generated_root=generated_root,
        test_manifest=test_manifest,
        test_root=test_root,
    )
    result.write(output, git_revision="abc123")

    assert (output / "per_sample.csv").is_file()
    assert (output / "summary.csv").is_file()
    payload = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert payload["git_revision"] == "abc123"
    assert payload["metric"]["same_species_only"] is True
    assert payload["metric"]["copy_risk_warning"] is True


def test_standardized_test_cache_is_converted_with_recorded_train_stats(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    project_root = tmp_path / "project"
    _write_array(generated_root / "A" / "g0.npy", 1.0)
    test_path = project_root / "artifacts" / "spectrograms" / "test" / "a0.npy"
    _write_array(test_path, 0.0)

    generated_manifest = tmp_path / "generated.csv"
    generated_manifest.write_text(
        "species,relative_path\nA,A/g0.npy\n", encoding="utf-8"
    )
    test_manifest = tmp_path / "spectrogram_test.csv"
    test_manifest.write_text(
        "split,name,species,relative_spec_path\ntest,A,alpha,artifacts/spectrograms/test/a0.npy\n",
        encoding="utf-8",
    )
    stats = tmp_path / "normalization_stats.json"
    stats.write_text(json.dumps({"mean_db": -50.0, "std_db": 10.0}) + "\n", encoding="utf-8")

    result = evaluate_generated_to_test(
        generated_manifest=generated_manifest,
        generated_root=generated_root,
        test_manifest=test_manifest,
        test_root=project_root,
        test_input_domain="standardized_logmel",
        test_normalization_stats=stats,
    )

    assert result.per_sample["generated_to_test"].iloc[0]["mse"] == pytest.approx(0.0)
    assert result.test_input_domain == "standardized_logmel"
    result.write(tmp_path / "report", git_revision="abc123")
    payload = json.loads((tmp_path / "report" / "protocol.json").read_text(encoding="utf-8"))
    assert payload["input_contract"]["test_source_domain"] == "standardized_logmel"
    assert payload["input_contract"]["test_conversion"].endswith("classifier_scale_from_standardized")
    assert payload["inputs"]["test_normalization_stats_sha256"]


def test_vae_v3_standardized_domain_compares_generated_spec_path_directly(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated"
    test_root = tmp_path / "test"
    generated_path = generated_root / "american_robin" / "g0.npy"
    _write_array(generated_path, 2.0)
    _write_array(test_root / "robin0.npy", 2.0)
    _write_array(test_root / "robin1.npy", 3.0)

    generated_manifest = tmp_path / "generated_v3.csv"
    generated_manifest.write_text(
        "name,spec_path\n"
        f"American Robin,{generated_path.resolve()}\n",
        encoding="utf-8",
    )
    test_manifest = tmp_path / "test_v3.csv"
    test_manifest.write_text(
        "split,name,relative_spec_path\n"
        "test,American Robin,robin0.npy\n"
        "test,American Robin,robin1.npy\n",
        encoding="utf-8",
    )

    result = evaluate_generated_to_test(
        generated_manifest=generated_manifest,
        generated_root=generated_root,
        test_manifest=test_manifest,
        test_root=test_root,
        comparison_domain="standardized_logmel",
        generated_input_domain="standardized_logmel",
        test_input_domain="standardized_logmel",
    )

    assert result.per_sample["generated_to_test"].iloc[0]["mse"] == pytest.approx(0.0)
    result.write(tmp_path / "report", git_revision="abc123")
    payload = json.loads((tmp_path / "report" / "protocol.json").read_text(encoding="utf-8"))
    assert payload["input_contract"]["comparison_domain"] == "standardized_logmel"
    assert payload["input_contract"]["numeric_domain"] == "standardized_logmel_float32"
    assert payload["input_contract"]["generated_conversion"] is None
    assert payload["input_contract"]["test_conversion"] is None
