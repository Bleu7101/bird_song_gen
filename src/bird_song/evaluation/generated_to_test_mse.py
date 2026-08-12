"""Same-species generated-to-test and real-to-real nearest-neighbour MSE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bird_song.generation.checkpoint_models import classifier_scale_from_standardized
from bird_song.evaluation.provenance import sha256_file


REQUIRED_COLUMNS = {"species", "relative_path"}
PROTOCOLS = ("generated_to_test", "real_to_real")
INPUT_DOMAINS = ("classifier_input", "standardized_logmel")


def _resolve_manifest_path(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} manifest contains a path outside its root: {value}")
    return path


def _load_input_array(
    path: Path,
    allow_project_cache_2d: bool,
    source_domain: str,
    comparison_domain: str,
    normalization: tuple[float, float] | None = None,
) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if allow_project_cache_2d and array.shape == (128, 128):
        array = array[None, ...]
    if array.shape != (1, 128, 128):
        raise ValueError(f"classifier input must have shape [1,128,128], got {array.shape} at {path}")
    if array.dtype.kind not in "fc" or not np.isfinite(array).all():
        raise ValueError(f"input must be finite floating point at {path}")
    if source_domain not in INPUT_DOMAINS:
        raise ValueError(f"unknown source input domain: {source_domain}")
    if comparison_domain not in INPUT_DOMAINS:
        raise ValueError(f"unknown comparison domain: {comparison_domain}")

    if source_domain == "classifier_input":
        if float(array.min()) < -1.00001 or float(array.max()) > 1.00001:
            raise ValueError(f"classifier input must be in [-1,1] at {path}")
        if comparison_domain == "standardized_logmel":
            raise ValueError(
                "classifier_input cannot be converted back to standardized_logmel "
                "because the relative-dB projection is not invertible"
            )
        converted = array
    elif comparison_domain == "standardized_logmel":
        converted = array
    else:
        if normalization is None:
            raise ValueError(
                "standardized_logmel input requires train normalization mean and std "
                "when comparison_domain=classifier_input"
            )
        mean, std = normalization
        converted = (
            classifier_scale_from_standardized(
                torch.from_numpy(array), mean=mean, std=std
            )
            .numpy()
            .astype(np.float32, copy=False)
        )

    if comparison_domain == "classifier_input" and (
        float(converted.min()) < -1.00001 or float(converted.max()) > 1.00001
    ):
        raise ValueError(f"classifier input must be in [-1,1] at {path}")
    return converted.astype(np.float32, copy=False)


def _read_manifest(
    path: Path,
    root: Path,
    label: str,
    allow_project_cache_2d: bool,
    split: str | None = None,
    input_domain: str = "classifier_input",
    comparison_domain: str = "classifier_input",
    normalization: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    frame = pd.read_csv(path)
    if split is not None and "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == split].copy()
    if "name" in frame.columns:
        # Project cache manifests also contain a Latin ``species`` field;
        # matching must use the common-name ``name`` field used by generators.
        frame["species"] = frame["name"]
    if "relative_path" not in frame.columns and "relative_spectrogram_path" in frame.columns:
        frame = frame.rename(columns={"relative_spectrogram_path": "relative_path"})
    if "relative_path" not in frame.columns and "relative_spec_path" in frame.columns:
        frame = frame.rename(columns={"relative_spec_path": "relative_path"})
    if "relative_path" not in frame.columns and "spec_path" in frame.columns:
        frame = frame.rename(columns={"spec_path": "relative_path"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{label} manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} manifest is empty: {path}")
    frame = frame.copy()
    frame["species"] = frame["species"].astype(str)
    frame["relative_path"] = frame["relative_path"].astype(str)
    paths = [_resolve_manifest_path(root, value, label) for value in frame["relative_path"]]
    arrays = []
    for item in paths:
        if not item.is_file():
            raise FileNotFoundError(f"{label} spectrogram is missing: {item}")
        arrays.append(
            _load_input_array(
                item,
                allow_project_cache_2d,
                source_domain=input_domain,
                comparison_domain=comparison_domain,
                normalization=normalization,
            )
        )
    return frame, arrays


def _read_normalization_stats(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        mean = float(payload["mean_db"])
        std = float(payload["std_db"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"normalization stats must contain numeric mean_db and std_db: {path}") from error
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"normalization stats must contain finite mean_db and positive std_db: {path}")
    return mean, std


def _nearest(source: np.ndarray, targets: list[np.ndarray], excluded: int | None = None) -> tuple[float, int]:
    candidates = [index for index in range(len(targets)) if index != excluded]
    if not candidates:
        return float("nan"), -1
    values = [float(np.mean((source - targets[index]) ** 2, dtype=np.float64)) for index in candidates]
    best = int(np.argmin(values))
    return values[best], candidates[best]


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["protocol", "species", "count", "mean", "std", "median", "q1", "q3", "min", "max"]
    output: list[dict[str, Any]] = []
    species_names = sorted(rows["species"].astype(str).unique())
    for protocol in PROTOCOLS:
        subset = rows[(rows["protocol"] == protocol) & rows["mse"].notna()]
        groups = [("overall", subset)] + [
            (species, subset[subset["species"] == species]) for species in species_names
        ]
        for species, group in groups:
            values = group["mse"]
            output.append(
                {
                    "protocol": protocol,
                    "species": species,
                    "count": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std()) if len(values) > 1 else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "q1": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q3": float(values.quantile(0.75)) if len(values) else np.nan,
                    "min": float(values.min()) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(output, columns=columns)


@dataclass
class EvaluationResult:
    per_sample: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    generated_manifest: Path
    test_manifest: Path
    generated_root: Path
    test_root: Path
    generated_metadata: Path | None = None
    comparison_domain: str = "classifier_input"
    generated_input_domain: str = "classifier_input"
    generated_normalization_stats: Path | None = None
    test_input_domain: str = "classifier_input"
    test_normalization_stats: Path | None = None

    def write(self, output: Path, git_revision: str = "unknown") -> None:
        output.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(self.per_sample.values(), ignore_index=True)
        combined.to_csv(output / "per_sample.csv", index=False)
        self.summary.to_csv(output / "summary.csv", index=False)
        provenance = {
            "schema_version": 1,
            "git_revision": git_revision,
            "inputs": {
                "generated_manifest": str(self.generated_manifest.resolve()),
                "generated_manifest_sha256": sha256_file(self.generated_manifest),
                "test_manifest": str(self.test_manifest.resolve()),
                "test_manifest_sha256": sha256_file(self.test_manifest),
                "generated_root": str(self.generated_root.resolve()),
                "test_root": str(self.test_root.resolve()),
                "generated_metadata": str(self.generated_metadata.resolve()) if self.generated_metadata else None,
                "generated_metadata_sha256": sha256_file(self.generated_metadata) if self.generated_metadata else None,
                "generated_normalization_stats": str(self.generated_normalization_stats.resolve()) if self.generated_normalization_stats else None,
                "generated_normalization_stats_sha256": sha256_file(self.generated_normalization_stats) if self.generated_normalization_stats else None,
                "test_normalization_stats": str(self.test_normalization_stats.resolve()) if self.test_normalization_stats else None,
                "test_normalization_stats_sha256": sha256_file(self.test_normalization_stats) if self.test_normalization_stats else None,
            },
            "input_contract": {
                "shape_after_load": [1, 128, 128],
                "comparison_domain": self.comparison_domain,
                "numeric_domain": (
                    "classifier_input_float32_[minus1,plus1]"
                    if self.comparison_domain == "classifier_input"
                    else "standardized_logmel_float32"
                ),
                "generated_source_domain": self.generated_input_domain,
                "test_source_domain": self.test_input_domain,
                "generated_conversion": (
                    "bird_song.generation.checkpoint_models.classifier_scale_from_standardized"
                    if self.generated_input_domain == "standardized_logmel"
                    and self.comparison_domain == "classifier_input" else None
                ),
                "test_conversion": (
                    "bird_song.generation.checkpoint_models.classifier_scale_from_standardized"
                    if self.test_input_domain == "standardized_logmel"
                    and self.comparison_domain == "classifier_input" else None
                ),
                "accepted_project_cache_2d_arrays": "expanded to leading singleton channel",
            },
            "metric": {
                "generated_to_test": "each generated sample to nearest same-species test spectrogram",
                "real_to_real": "each test sample to nearest other same-species test spectrogram",
                "same_species_only": True,
                "copy_risk_warning": True,
                "interpretation": "pixel-space proximity only; low MSE can reward copying and is not standalone perceptual realism",
            },
            "test_set_policy": "test is used only for this final frozen evaluation; no selection, tuning, posterior-bank fitting, or checkpoint choice",
        }
        (output / "protocol.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def evaluate_generated_to_test(
    *, generated_manifest: Path, generated_root: Path, test_manifest: Path, test_root: Path,
    generated_metadata: Path | None = None,
    comparison_domain: str = "classifier_input",
    generated_input_domain: str = "classifier_input",
    generated_normalization_stats: Path | None = None,
    test_input_domain: str = "classifier_input",
    test_normalization_stats: Path | None = None,
) -> EvaluationResult:
    generated_manifest = generated_manifest.resolve()
    generated_root = generated_root.resolve()
    test_manifest = test_manifest.resolve()
    test_root = test_root.resolve()
    for label, domain in (
        ("comparison", comparison_domain),
        ("generated input", generated_input_domain),
        ("test input", test_input_domain),
    ):
        if domain not in INPUT_DOMAINS:
            raise ValueError(f"unknown {label} domain: {domain}")
    if generated_input_domain == "classifier_input" and comparison_domain == "standardized_logmel":
        raise ValueError(
            "classifier_input generated arrays cannot be compared in standardized_logmel "
            "because the relative-dB projection is not invertible"
        )
    if test_input_domain == "classifier_input" and comparison_domain == "standardized_logmel":
        raise ValueError(
            "classifier_input test arrays cannot be compared in standardized_logmel "
            "because the relative-dB projection is not invertible"
        )
    if generated_normalization_stats is not None:
        generated_normalization_stats = generated_normalization_stats.resolve()
        if not generated_normalization_stats.is_file():
            raise FileNotFoundError(
                f"generated normalization stats are missing: {generated_normalization_stats}"
            )
    if test_normalization_stats is not None:
        test_normalization_stats = test_normalization_stats.resolve()
        if not test_normalization_stats.is_file():
            raise FileNotFoundError(f"test normalization stats are missing: {test_normalization_stats}")
    generated_normalization = (
        _read_normalization_stats(generated_normalization_stats)
        if generated_input_domain == "standardized_logmel"
        and comparison_domain == "classifier_input"
        and generated_normalization_stats is not None else None
    )
    if generated_input_domain == "standardized_logmel" and comparison_domain == "classifier_input" and generated_normalization is None:
        raise ValueError(
            "standardized_logmel generated input requires --generated-normalization-stats "
            "when comparison_domain=classifier_input"
        )
    test_normalization = (
        _read_normalization_stats(test_normalization_stats)
        if test_input_domain == "standardized_logmel"
        and comparison_domain == "classifier_input"
        and test_normalization_stats is not None else None
    )
    if test_input_domain == "standardized_logmel" and comparison_domain == "classifier_input" and test_normalization is None:
        raise ValueError(
            "standardized_logmel test input requires --test-normalization-stats "
            "when comparison_domain=classifier_input"
        )
    generated, generated_arrays = _read_manifest(
        generated_manifest,
        generated_root,
        "generated",
        False,
        input_domain=generated_input_domain,
        comparison_domain=comparison_domain,
        normalization=generated_normalization,
    )
    test, test_arrays = _read_manifest(
        test_manifest,
        test_root,
        "test",
        True,
        split="test",
        input_domain=test_input_domain,
        comparison_domain=comparison_domain,
        normalization=test_normalization,
    )
    generated_rows: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    for row, array in zip(generated.to_dict("records"), generated_arrays, strict=True):
        candidates = [i for i, value in enumerate(test["species"]) if value == row["species"]]
        mse, best_local = _nearest(array, [test_arrays[i] for i in candidates]) if candidates else (np.nan, -1)
        generated_rows.append({
            "protocol": "generated_to_test",
            "species": row["species"],
            "source_path": row["relative_path"],
            "nearest_target_path": test.iloc[candidates[best_local]]["relative_path"] if best_local >= 0 else "",
            "mse": mse,
        })
    for index, (row, array) in enumerate(zip(test.to_dict("records"), test_arrays, strict=True)):
        candidates = [i for i, value in enumerate(test["species"]) if value == row["species"]]
        mse, target = _nearest(array, [test_arrays[i] for i in candidates], candidates.index(index) if index in candidates else None)
        real_rows.append({
            "protocol": "real_to_real",
            "species": row["species"],
            "source_path": row["relative_path"],
            "nearest_target_path": test.iloc[candidates[target]]["relative_path"] if target >= 0 else "",
            "mse": mse,
        })
    per_sample = {name: frame for name, frame in pd.concat([pd.DataFrame(generated_rows), pd.DataFrame(real_rows)]).groupby("protocol", sort=False)}
    combined = pd.concat(per_sample.values(), ignore_index=True)
    if generated_metadata is not None:
        generated_metadata = generated_metadata.resolve()
        if not generated_metadata.is_file():
            raise FileNotFoundError(f"generated metadata is missing: {generated_metadata}")
    return EvaluationResult(
        per_sample,
        _summary(combined),
        generated_manifest,
        test_manifest,
        generated_root,
        test_root,
        generated_metadata,
        comparison_domain,
        generated_input_domain,
        generated_normalization_stats,
        test_input_domain,
        test_normalization_stats,
    )
