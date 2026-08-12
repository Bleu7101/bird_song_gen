"""Notebook-compatible, frozen VAE-v3 evaluation-pool helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bird_song.evaluation.provenance import sha256_file


GENERATION_CELL_INDICES = (2, 4, 5, 7, 9, 10, 12, 14, 16, 18, 20, 24, 26)
GENERATION_CELL_INDEX = 34
GENERATION_CALL_MARKER = "posterior_bank = fit_class_conditional_posterior_bank"
TEST_SMOKE_MARKER = "test_loss, test_parts = vae_loss("


def ensure_fresh_output(path: Path) -> None:
    """Create an output directory only when it is absent or empty."""
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output exists but is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)


def select_generation_cells(notebook: Any) -> Any:
    """Select notebook cells needed for model loading and generation only."""
    import nbformat

    cells = []
    for index in GENERATION_CELL_INDICES:
        cell = copy.deepcopy(notebook.cells[index])
        if index == 24 and TEST_SMOKE_MARKER in str(cell.source):
            cell.source = str(cell.source).split(TEST_SMOKE_MARKER, 1)[0].rstrip()
        cells.append(cell)
    source = str(notebook.cells[GENERATION_CELL_INDEX].source)
    if GENERATION_CALL_MARKER not in source:
        raise ValueError(
            "04 notebook generation cell no longer contains the expected posterior-bank call"
        )
    definitions = source.split(GENERATION_CALL_MARKER, 1)[0].rstrip()
    cells.append(nbformat.v4.new_code_cell(definitions))
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata=copy.deepcopy(getattr(notebook, "metadata", {})),
        nbformat=notebook.nbformat,
        nbformat_minor=notebook.nbformat_minor,
    )


def _quoted_path(path: Path) -> str:
    return json.dumps(str(path.resolve()))


def build_generation_notebook(
    notebook: Any,
    *,
    project_root: Path,
    checkpoint: Path,
    posterior_bank: Path,
    output: Path,
    samples_per_species: int,
    seed: int,
    temperature: float,
    device: str,
) -> Any:
    """Build an executed-notebook input with explicit, non-default paths."""
    import nbformat

    if samples_per_species < 1:
        raise ValueError("samples_per_species must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    selected = select_generation_cells(notebook)
    config_cell = selected.cells[1]
    if device == "auto":
        device_expression = 'torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    elif device in {"cpu", "cuda", "mps"}:
        device_expression = f"torch.device({json.dumps(device)})"
    else:
        raise ValueError(f"unsupported device: {device}")
    _, class_and_body = str(config_cell.source).split("@dataclass", 1)
    config_cell.source = (
        "from pathlib import Path\n"
        f"PROJECT_ROOT = Path({_quoted_path(project_root)}).resolve()\n"
        "if str(PROJECT_ROOT / \"src\") not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n\n"
        "@dataclass"
        + class_and_body
    )
    override = f"""

# Codex evaluation-pool override: inputs remain in the original project;
# generated outputs are redirected to a fresh versioned run directory.
cfg.PROJECT_ROOT = str(PROJECT_ROOT)
cfg.CHECKPOINT_DIR = {_quoted_path(checkpoint.parent)}
cfg.OUTPUT_DIR = {_quoted_path(output)}
cfg.SAMPLES_PER_SPECIES = {int(samples_per_species)}
cfg.PRIOR_TEMPERATURE = {float(temperature)!r}
cfg.SEED = {int(seed)}
cfg.RUN_TRAINING = False
cfg.RESUME_FROM_LAST = False
DEVICE = {device_expression}
AMP_ENABLED = bool(cfg.AMP and DEVICE.type == "cuda")
"""
    anchor = "cfg = Config()\n"
    if anchor not in config_cell.source:
        raise ValueError("04 notebook config cell no longer contains cfg = Config()")
    config_cell.source = config_cell.source.replace(anchor, anchor + override, 1)

    bank_cell = nbformat.v4.new_code_cell(
        f"""# Load the existing train-only posterior bank; do not refit it from test data.
existing_posterior_bank_path = Path({_quoted_path(posterior_bank)})
posterior_package = torch.load(
    existing_posterior_bank_path, map_location="cpu", weights_only=True
)
assert posterior_package.get("fitted_split") == "train"
assert abs(float(posterior_package.get("temperature", -1.0)) - cfg.PRIOR_TEMPERATURE) < 1e-8
assert dict(posterior_package.get("label_to_id", {{}})) == label_to_id
posterior_bank = {{
    int(label_id): {{
        "mu": value["mu"].float(),
        "logvar": value["logvar"].float(),
        "paths": list(value["paths"]),
    }}
    for label_id, value in posterior_package["banks"].items()
}}
generated_samples, generated_labels = generate_and_save_samples(
    model, cfg.SAMPLES_PER_SPECIES, posterior_bank
)
print("Generated samples:", tuple(generated_samples.shape))
print("Generated labels:", tuple(generated_labels.shape))
"""
    )
    selected.cells.append(bank_cell)
    return selected


def _relative_output_path(value: str, output_root: Path) -> str:
    path = Path(value).resolve()
    root = output_root.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"generated path is outside pool root: {path}")
    return path.relative_to(root).as_posix()


def make_portable_manifest(frame: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Replace notebook absolute generated paths with pool-relative paths."""
    required = {"spec_path", "classifier_spec_path", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"generated manifest is missing columns: {sorted(missing)}")
    output = frame.copy()
    output["relative_path"] = output["spec_path"].map(
        lambda value: _relative_output_path(str(value), output_root)
    )
    output["relative_classifier_path"] = output["classifier_spec_path"].map(
        lambda value: _relative_output_path(str(value), output_root)
    )
    output["species"] = output["name"].astype(str)
    return output.drop(columns=["spec_path", "classifier_spec_path"])


def build_array_hashes(frame: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        for kind, column in (
            ("standardized_logmel", "relative_path"),
            ("classifier_input", "relative_classifier_path"),
        ):
            relative = str(record[column])
            path = (output_root / relative).resolve()
            if not path.is_relative_to(output_root.resolve()) or not path.is_file():
                raise FileNotFoundError(f"pool array is missing: {path}")
            array = np.load(path, allow_pickle=False)
            if array.shape != (1, 128, 128):
                raise ValueError(f"pool array has unexpected shape {array.shape}: {path}")
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"pool array must be finite float32: {path}")
            rows.append(
                {
                    "kind": kind,
                    "species": str(record["species"]),
                    "relative_path": relative,
                    "sha256": sha256_file(path),
                    "shape": "1x128x128",
                    "dtype": str(array.dtype),
                    "min": float(array.min()),
                    "max": float(array.max()),
                }
            )
    return pd.DataFrame(rows)


def build_pool_metadata(
    *,
    notebook: Path,
    project_root: Path,
    checkpoint: Path,
    posterior_bank: Path,
    output: Path,
    notebook_sha256: str,
    checkpoint_sha256: str,
    posterior_bank_sha256: str,
    git_revision: str,
    seed: int,
    generation_stream_seed: int,
    samples_per_species: int,
    temperature: float,
    device_used: str,
    label_to_id: Mapping[str, int],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generator": "spatial_detail_conditional_vae_v3_posterior_mixture",
        "source_notebook": str(notebook.resolve()),
        "project_root": str(project_root.resolve()),
        "output_root": str(output.resolve()),
        "seed": int(seed),
        "generation_stream_seed": int(generation_stream_seed),
        "samples_per_species": int(samples_per_species),
        "total_samples": int(samples_per_species * len(label_to_id)),
        "temperature": float(temperature),
        "device_used": str(device_used),
        "class_order": list(label_to_id),
        "label_to_id": {str(key): int(value) for key, value in label_to_id.items()},
        "comparison_domain": "standardized_logmel",
        "array_contract": {
            "shape": [1, 128, 128],
            "dtype": "float32",
            "generated_domain": "global_train_standardized_logmel",
            "auxiliary_domain": "relative_db_classifier_input_minus1_plus1",
        },
        "sampling": {
            "anchor_source": "existing posterior bank",
            "posterior_bank_fitted_split": "train",
            "random_stream": "torch.Generator(device=DEVICE.type).manual_seed(seed + 100)",
            "latent_formula": "mu + temperature * exp(0.5 * logvar) * epsilon",
            "notebook_function": "generate_and_save_samples",
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha256,
        },
        "posterior_bank": {
            "path": str(posterior_bank.resolve()),
            "sha256": posterior_bank_sha256,
            "fitted_split": "train",
            "temperature": float(temperature),
        },
        "provenance": {
            "git_revision": git_revision,
            "source_notebook_sha256": notebook_sha256,
            "test_set_used_for_generation": False,
            "selection_or_tuning_on_test": False,
        },
    }
