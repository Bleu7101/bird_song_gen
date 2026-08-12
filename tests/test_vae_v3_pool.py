from __future__ import annotations

from pathlib import Path

import nbformat
import pandas as pd
import pytest

from bird_song.evaluation.vae_v3_pool import (
    build_generation_notebook,
    build_pool_metadata,
    ensure_fresh_output,
    make_portable_manifest,
    select_generation_cells,
)


def _synthetic_notebook():
    cells = [nbformat.v4.new_code_cell(f"cell_{index}") for index in range(35)]
    cells[4] = nbformat.v4.new_code_cell(
        "PROJECT_ROOT = Path.cwd().resolve()\n"
        "@dataclass\n"
        "class Config:\n"
        "    pass\n\n"
        "cfg = Config()\n"
    )
    cells[24] = nbformat.v4.new_code_cell(
        "def make_grad_scaler():\n"
        "    return None\n\n"
        "test_loss, test_parts = vae_loss()\n"
    )
    cells[34] = nbformat.v4.new_code_cell(
        "def generate_and_save_samples():\n"
        "    return None\n\n"
        "posterior_bank = fit_class_conditional_posterior_bank()\n"
        "generated_samples = generate_and_save_samples()\n"
    )
    return nbformat.v4.new_notebook(cells=cells)


def test_ensure_fresh_output_rejects_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "pool"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        ensure_fresh_output(output)


def test_portable_manifest_uses_paths_relative_to_pool_root(tmp_path: Path) -> None:
    output = tmp_path / "pool"
    generated = output / "generated_npy" / "american_robin" / "000.npy"
    classifier = output / "classifier_input" / "american_robin" / "000.npy"
    generated.parent.mkdir(parents=True)
    classifier.parent.mkdir(parents=True)
    generated.write_bytes(b"generated")
    classifier.write_bytes(b"classifier")
    frame = pd.DataFrame(
        [
            {
                "filename": "000.npy",
                "spec_path": str(generated.resolve()),
                "classifier_spec_path": str(classifier.resolve()),
                "name": "American Robin",
                "label_id": 2,
                "source_model": "spatial_detail_conditional_vae_v3_posterior_mixture",
                "anchor_spec_path": r"C:\\train\\anchor.npy",
                "posterior_temperature": 0.35,
            }
        ]
    )

    portable = make_portable_manifest(frame, output)

    assert portable.loc[0, "relative_path"] == "generated_npy/american_robin/000.npy"
    assert portable.loc[0, "relative_classifier_path"] == "classifier_input/american_robin/000.npy"
    assert portable.loc[0, "name"] == "American Robin"


def test_generation_notebook_selects_definitions_and_injects_existing_bank(tmp_path: Path) -> None:
    notebook = _synthetic_notebook()
    selected = select_generation_cells(notebook)
    selected_source = "\n".join(cell.source for cell in selected.cells)

    assert "cell_2" in selected_source
    assert "cell_26" in selected_source
    assert "posterior_bank = fit_class_conditional_posterior_bank" not in selected_source
    assert "test_loss, test_parts = vae_loss" not in selected_source

    configured = build_generation_notebook(
        notebook,
        project_root=tmp_path / "project",
        checkpoint=tmp_path / "checkpoint.pt",
        posterior_bank=tmp_path / "bank.pt",
        output=tmp_path / "pool",
        samples_per_species=16,
        seed=42,
        temperature=0.35,
        device="cpu",
    )
    configured_source = "\n".join(cell.source for cell in configured.cells)

    assert "existing_posterior_bank_path" in configured_source
    assert "cfg.SAMPLES_PER_SPECIES = 16" in configured_source
    assert "cfg.PRIOR_TEMPERATURE = 0.35" in configured_source
    assert "torch.device(\"cpu\")" in configured_source
    assert "test_metrics = run_epoch" not in configured_source


def test_pool_metadata_records_frozen_protocol(tmp_path: Path) -> None:
    metadata = build_pool_metadata(
        notebook=tmp_path / "04.ipynb",
        project_root=tmp_path / "project",
        checkpoint=tmp_path / "checkpoint.pt",
        posterior_bank=tmp_path / "bank.pt",
        output=tmp_path / "pool",
        notebook_sha256="notebook-hash",
        checkpoint_sha256="checkpoint-hash",
        posterior_bank_sha256="bank-hash",
        git_revision="abc123",
        seed=42,
        generation_stream_seed=142,
        samples_per_species=16,
        temperature=0.35,
        device_used="cuda",
        label_to_id={"Northern Cardinal": 0, "Song Sparrow": 1, "American Robin": 2},
    )

    assert metadata["seed"] == 42
    assert metadata["generation_stream_seed"] == 142
    assert metadata["samples_per_species"] == 16
    assert metadata["temperature"] == pytest.approx(0.35)
    assert metadata["device_used"] == "cuda"
    assert metadata["comparison_domain"] == "standardized_logmel"
    assert metadata["posterior_bank"]["fitted_split"] == "train"
    assert metadata["provenance"]["git_revision"] == "abc123"
