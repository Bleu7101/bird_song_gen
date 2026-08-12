"""Deterministically filter an existing VAE posterior bank to content-safe training anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from .checkpoint_models import GENERATOR_CLASSES, VAE_TEMPERATURE


POSTERIOR_BANK_SCHEMA_VERSION = 2
POSTERIOR_BANK_CONTRACT = "content_safe_v2_train_filtered_existing_bank_v1"
POSTERIOR_BANK_SOURCE_MANIFEST = "manifests/content_safe_v2/full_dataset_train.csv"
POSTERIOR_BANK_EXPECTED_COUNTS = {
    "Northern Cardinal": 256,
    "Song Sparrow": 247,
    "American Robin": 256,
}
POSTERIOR_BANK_EXPECTED_REMOVALS = {
    "Northern Cardinal": 0,
    "Song Sparrow": 9,
    "American Robin": 0,
}


def _anchor_stem(path: object) -> str:
    return Path(str(path).replace("\\", "/")).stem


def filter_existing_posterior_bank(
    bank_package: Mapping[str, Any],
    train_manifest: pd.DataFrame,
    *,
    enforce_canonical_counts: bool = True,
) -> dict[str, Any]:
    """Preserve existing posterior tensors whose source occurs uniquely in the train manifest."""
    required_columns = {"name", "filename", "relative_wav_path", "split"}
    missing = required_columns - set(train_manifest.columns)
    if missing:
        raise ValueError(f"training manifest is missing columns: {sorted(missing)}")
    if set(train_manifest["split"].astype(str)) != {"train"}:
        raise ValueError("posterior-bank filter requires a train-only manifest")

    label_to_id = dict(bank_package.get("label_to_id", {}))
    expected_label_to_id = {name: index for index, name in enumerate(GENERATOR_CLASSES)}
    if label_to_id != expected_label_to_id:
        raise ValueError(f"posterior-bank label map mismatch: {label_to_id}")
    if bank_package.get("fitted_split") != "train":
        raise ValueError("posterior bank must have been fitted on the training split")
    if abs(float(bank_package.get("temperature", -1.0)) - VAE_TEMPERATURE) > 1e-8:
        raise ValueError("posterior-bank temperature mismatch")
    source_banks = bank_package.get("banks")
    if not isinstance(source_banks, Mapping):
        raise ValueError("posterior bank is missing banks")

    manifest_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in train_manifest.to_dict(orient="records"):
        key = (str(row["name"]), _anchor_stem(row["filename"]))
        manifest_rows.setdefault(key, []).append(row)

    filtered_banks: dict[int, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    original_counts: dict[str, int] = {}
    removed_counts: dict[str, int] = {}
    removed_anchors: dict[str, list[dict[str, Any]]] = {}
    for species, label_id in expected_label_to_id.items():
        raw = source_banks.get(label_id, source_banks.get(str(label_id)))
        if not isinstance(raw, Mapping):
            raise ValueError(f"posterior bank is missing label {label_id}")
        mu = torch.as_tensor(raw.get("mu"), dtype=torch.float32, device="cpu")
        logvar = torch.as_tensor(raw.get("logvar"), dtype=torch.float32, device="cpu")
        paths = list(raw.get("paths", []))
        if mu.ndim != 4 or tuple(mu.shape[1:]) != (16, 16, 16):
            raise ValueError(f"posterior-bank latent shape mismatch for {species}: {tuple(mu.shape)}")
        if logvar.shape != mu.shape or len(paths) != len(mu) or len(mu) == 0:
            raise ValueError(f"posterior-bank entries are inconsistent for {species}")

        kept_indices: list[int] = []
        relative_wav_paths: list[str] = []
        removed: list[dict[str, Any]] = []
        for source_index, path in enumerate(paths):
            key = (species, _anchor_stem(path))
            matches = manifest_rows.get(key, [])
            if len(matches) > 1:
                raise ValueError(f"posterior anchor is not unique in training manifest: {key}")
            if not matches:
                removed.append({"source_index": source_index, "source_anchor_name": key[1]})
                continue
            kept_indices.append(source_index)
            relative_wav_paths.append(str(matches[0]["relative_wav_path"]).replace("\\", "/"))

        index_tensor = torch.tensor(kept_indices, dtype=torch.long)
        filtered_banks[label_id] = {
            "mu": mu.index_select(0, index_tensor),
            "logvar": logvar.index_select(0, index_tensor),
            "paths": list(relative_wav_paths),
            "source_indices": list(kept_indices),
            "relative_wav_paths": list(relative_wav_paths),
        }
        original_counts[species] = len(paths)
        counts[species] = len(kept_indices)
        removed_counts[species] = len(removed)
        removed_anchors[species] = removed

    if enforce_canonical_counts:
        if counts != POSTERIOR_BANK_EXPECTED_COUNTS:
            raise ValueError(f"unexpected filtered posterior-bank counts: {counts}")
        if removed_counts != POSTERIOR_BANK_EXPECTED_REMOVALS:
            raise ValueError(f"unexpected posterior-bank removals: {removed_counts}")

    return {
        "schema_version": POSTERIOR_BANK_SCHEMA_VERSION,
        "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
        "source_manifest": POSTERIOR_BANK_SOURCE_MANIFEST,
        "derivation": "filtered_existing_posterior_bank",
        "vae_checkpoint_retrained": False,
        "fitted_split": "train",
        "temperature": VAE_TEMPERATURE,
        "sampling_type": bank_package.get(
            "sampling_type", "per_species_posterior_anchor_mixture"
        ),
        "label_to_id": label_to_id,
        "original_counts": original_counts,
        "counts": counts,
        "removed_counts": removed_counts,
        "removed_anchors": removed_anchors,
        "banks": filtered_banks,
    }


def filter_posterior_bank_file(input_path: Path, manifest_path: Path, output_path: Path) -> None:
    """Load, filter, and atomically publish a bank without modifying the input."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("input and output posterior-bank paths must differ")
    package = torch.load(input_path, map_location="cpu", weights_only=True)
    manifest = pd.read_csv(manifest_path)
    filtered = filter_existing_posterior_bank(package, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(filtered, temporary)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
