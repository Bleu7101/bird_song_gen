"""Frozen-checkpoint generator evaluation and bounded report packaging.

The metrics in this module are deliberately classifier-view diagnostics.  They
measure conditioning compatibility, feature-space similarity, diversity,
coverage, copy risk, and seed stability; they do not make waveform or human
realism claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, pairwise_distances, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from bird_song.audio import load_generated_spectrogram
from bird_song.config import DEFAULT_CLASSES, SpectrogramConfig
from bird_song.data import ManifestDataset, make_loader, resolve_spectrogram_cache_root
from bird_song.generation.checkpoint_pool import (
    EXPECTED_CHECKPOINT_SHA256,
    GENERATOR_CLASSES,
    _valid_classifier_array,
    sha256_array,
    sha256_file,
    species_slug,
    verify_checkpoint,
)
from bird_song.runtime import choose_device, load_checkpoint
from bird_song.spectrogram_cache import load_cache_array


CONTENT_SAFE_CLASSES = tuple(DEFAULT_CLASSES)
EVALUATION_SEEDS = (42, 123, 777)
SAMPLE_SIZE = 128
RESAMPLES = 200
PCA_COMPONENTS = 64
COPY_RISK_QUANTILE = 0.05
PRIMARY_CALIBRATION = {"accuracy": 0.8997955010224948, "macro_f1": 0.9016177383860278}
RESIDUAL_CALIBRATION = {"accuracy": 0.9038854805725971, "macro_f1": 0.9044463009375291}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass
class FrozenEvaluator:
    name: str
    checkpoint_path: Path
    model: torch.nn.Module
    classes: tuple[str, ...]
    config: SpectrogramConfig
    checkpoint: dict[str, Any]


class PoolDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, rows: pd.DataFrame, root: Path, classes: Sequence[str]):
        self.rows = rows.reset_index(drop=True)
        self.root = root.resolve()
        self.classes = tuple(classes)
        self.class_to_id = {name: index for index, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[index]
        path = (self.root / str(row["relative_path"])).resolve()
        array = _valid_classifier_array(path)
        if array is None:
            raise ValueError(f"invalid generated classifier array: {path}")
        label = self.class_to_id[str(row["species"])]
        return torch.from_numpy(np.array(array, copy=True)), label, str(path)


def load_evaluator(path: Path, device: torch.device, expected_name: str) -> FrozenEvaluator:
    model, classes, config, checkpoint = load_checkpoint(path.resolve(), device)
    model.eval()
    model.requires_grad_(False)
    if tuple(classes) != CONTENT_SAFE_CLASSES:
        raise ValueError(f"{expected_name} class order mismatch: {classes}")
    if expected_name == "crnn" and checkpoint.get("architecture", checkpoint.get("model_config", {}).get("architecture")) != "crnn":
        raise ValueError("primary evaluator is not a CRNN checkpoint")
    if expected_name == "residual" and checkpoint.get("architecture", "residual_cnn") not in {None, "residual_cnn"}:
        raise ValueError("sensitivity evaluator is not the legacy residual checkpoint")
    return FrozenEvaluator(expected_name, path.resolve(), model, tuple(classes), config, dict(checkpoint))


def _collect_outputs(
    evaluator: FrozenEvaluator,
    dataset: Dataset[tuple[torch.Tensor, int, str]],
    device: torch.device,
    batch_size: int = 128,
    include_arrays: bool = False,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_rows: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    arrays: list[np.ndarray] = []
    with torch.inference_mode():
        for specs, batch_labels, batch_paths in loader:
            specs = specs.to(device)
            features = evaluator.model.forward_features(specs).float().cpu().numpy()
            logits = evaluator.model(specs).float().cpu().numpy()
            feature_rows.append(features)
            logits_rows.append(logits)
            labels.extend(batch_labels.tolist())
            paths.extend(list(batch_paths))
            if include_arrays:
                arrays.extend(specs.cpu().numpy().astype(np.float32, copy=False))
    logits_array = np.concatenate(logits_rows, axis=0)
    logits_shifted = logits_array - logits_array.max(axis=1, keepdims=True)
    probabilities = np.exp(logits_shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "features": np.concatenate(feature_rows, axis=0),
        "logits": logits_array,
        "probabilities": probabilities,
        "labels": np.asarray(labels, dtype=np.int64),
        "paths": paths,
        "arrays": np.asarray(arrays, dtype=np.float32) if include_arrays else None,
    }


def _classification_metrics(outputs: Mapping[str, Any], classes: Sequence[str]) -> dict[str, Any]:
    labels = np.asarray(outputs["labels"])
    probabilities = np.asarray(outputs["probabilities"])
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(classes))))
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(len(classes))), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=list(range(len(classes))), average="macro", zero_division=0)),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_entropy": float((-probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1).mean()),
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions,
        "per_class_recall": recall_score(labels, predictions, labels=list(range(len(classes))), average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(labels, predictions, labels=list(range(len(classes))), average=None, zero_division=0).tolist(),
    }


def _fit_feature_projection(train_features: np.ndarray) -> tuple[StandardScaler, PCA]:
    scaler = StandardScaler().fit(train_features)
    standardized = scaler.transform(train_features)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=20260810).fit(standardized)
    return scaler, pca


def _project(features: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    return pca.transform(scaler.transform(features)).astype(np.float64, copy=False)


def _frechet_distance(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    mean_left = left.mean(axis=0)
    mean_right = right.mean(axis=0)
    cov_left = np.cov(left, rowvar=False)
    cov_right = np.cov(right, rowvar=False)
    # Symmetric eigen decomposition avoids a scipy dependency and is stable for
    # the small PCA covariance matrices used here.
    values, vectors = np.linalg.eigh((cov_left + cov_left.T) / 2.0)
    root_left = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
    middle = root_left @ cov_right @ root_left
    middle_values, middle_vectors = np.linalg.eigh((middle + middle.T) / 2.0)
    root_middle = (middle_vectors * np.sqrt(np.clip(middle_values, 0.0, None))) @ middle_vectors.T
    squared = float(np.sum((mean_left - mean_right) ** 2) + np.trace(cov_left + cov_right - 2.0 * root_middle))
    return max(0.0, squared)


def _manifold_metrics(real: np.ndarray, generated: np.ndarray, k: int = 5) -> dict[str, float]:
    if len(real) <= k or len(generated) <= k:
        return {"precision": float("nan"), "recall": float("nan"), "density": float("nan"), "coverage": float("nan")}
    real_real = pairwise_distances(real, real)
    gen_gen = pairwise_distances(generated, generated)
    real_gen = pairwise_distances(real, generated)
    real_radius = np.partition(real_real, k, axis=1)[:, k]
    gen_radius = np.partition(gen_gen, k, axis=1)[:, k]
    precision_hits = (real_gen <= real_radius[:, None]).any(axis=0)
    recall_hits = (real_gen <= gen_radius[None, :]).any(axis=1)
    density = (real_gen <= real_radius[:, None]).sum(axis=0).mean() / k
    coverage = (real_gen.min(axis=1) <= real_radius).mean()
    return {
        "precision": float(precision_hits.mean()),
        "recall": float(recall_hits.mean()),
        "density": float(density),
        "coverage": float(coverage),
    }


def _mean_pairwise_distance(features: np.ndarray) -> float:
    if len(features) < 2:
        return float("nan")
    distances = pairwise_distances(features)
    triangle = distances[np.triu_indices(len(features), 1)]
    return float(triangle.mean())


def _resampled_feature_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    seed: int,
    species_index: int,
    resamples: int = RESAMPLES,
    sample_size: int = SAMPLE_SIZE,
) -> dict[str, float]:
    rng = np.random.default_rng(20260810 + seed * 1009 + species_index * 97)
    values: dict[str, list[float]] = {"frechet": [], "precision": [], "recall": [], "density": [], "coverage": [], "generated_diversity": [], "real_diversity": [], "diversity_ratio": [], "diversity_delta": []}
    real_replace = len(real) < sample_size
    generated_replace = len(generated) < sample_size
    for _ in range(resamples):
        real_indices = rng.choice(len(real), sample_size, replace=real_replace)
        generated_indices = rng.choice(len(generated), sample_size, replace=generated_replace)
        real_sample = real[real_indices]
        generated_sample = generated[generated_indices]
        values["frechet"].append(_frechet_distance(real_sample, generated_sample))
        values["precision"].append(_manifold_metrics(real_sample, generated_sample)["precision"])
        values["recall"].append(_manifold_metrics(real_sample, generated_sample)["recall"])
        values["density"].append(_manifold_metrics(real_sample, generated_sample)["density"])
        values["coverage"].append(_manifold_metrics(real_sample, generated_sample)["coverage"])
        generated_diversity = _mean_pairwise_distance(generated_sample)
        real_diversity = _mean_pairwise_distance(real_sample)
        values["generated_diversity"].append(generated_diversity)
        values["real_diversity"].append(real_diversity)
        values["diversity_ratio"].append(generated_diversity / max(real_diversity, 1e-12))
        values["diversity_delta"].append(generated_diversity - real_diversity)
    output: dict[str, float] = {}
    for name, sample_values in values.items():
        array = np.asarray(sample_values, dtype=float)
        output[f"{name}_mean"] = float(np.mean(array))
        output[f"{name}_std"] = float(np.std(array, ddof=1))
        output[f"{name}_min"] = float(np.min(array))
        output[f"{name}_max"] = float(np.max(array))
    output["resamples"] = resamples
    output["sample_size"] = sample_size
    return output


def _image_metrics(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    def values(images: np.ndarray) -> dict[str, float]:
        time_gradient = np.abs(np.diff(images, axis=3)).mean(axis=(1, 2, 3))
        frequency_gradient = np.abs(np.diff(images, axis=2)).mean(axis=(1, 2, 3))
        dynamic_range = images.max(axis=(1, 2, 3)) - images.min(axis=(1, 2, 3))
        active_fraction = (images > -0.8).mean(axis=(1, 2, 3))
        return {
            "time_gradient_energy": float(time_gradient.mean()),
            "frequency_gradient_energy": float(frequency_gradient.mean()),
            "dynamic_range": float(dynamic_range.mean()),
            "active_bin_fraction": float(active_fraction.mean()),
            "pixel_mean": float(images.mean()),
            "pixel_std": float(images.std()),
        }

    real_values = values(real)
    generated_values = values(generated)
    quantiles = np.linspace(0.01, 0.99, 99)
    real_quantiles = np.quantile(real, quantiles)
    generated_quantiles = np.quantile(generated, quantiles)
    output = {f"real_{key}": value for key, value in real_values.items()}
    output.update({f"generated_{key}": value for key, value in generated_values.items()})
    output["pixel_distribution_wasserstein"] = float(np.mean(np.abs(real_quantiles - generated_quantiles)))
    output["pixel_mean_delta"] = generated_values["pixel_mean"] - real_values["pixel_mean"]
    output["pixel_std_delta"] = generated_values["pixel_std"] - real_values["pixel_std"]
    return output


def _pool_rows(pool_root: Path, model: str, seed: int, expected_samples: int = 200) -> tuple[pd.DataFrame, Path]:
    manifest = pool_root / model / f"seed_{seed}" / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"pool manifest missing: {manifest}")
    root = manifest.parent
    metadata_path = root / "generation.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("checkpoint_sha256", "")).upper() != EXPECTED_CHECKPOINT_SHA256[model]:
            raise ValueError(f"pool checkpoint provenance mismatch: {metadata_path}")
        if int(metadata.get("seed", seed)) != int(seed) or int(metadata.get("samples_per_class", expected_samples)) != expected_samples:
            raise ValueError(f"pool generation settings mismatch: {metadata_path}")
    frame = pd.read_csv(manifest)
    required = {"species", "relative_path", "generator", "pool_rank", "checkpoint_sha256", "array_sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pool manifest missing columns: {sorted(missing)}")
    if len(frame) != expected_samples * len(GENERATOR_CLASSES):
        raise ValueError(f"pool {manifest} is incomplete: {len(frame)} rows")
    counts = frame["species"].value_counts().to_dict()
    if any(int(counts.get(species, 0)) != expected_samples for species in GENERATOR_CLASSES):
        raise ValueError(f"pool {manifest} is not balanced by species")
    observed_hashes: list[str] = []
    for row in frame.to_dict(orient="records"):
        path = root / str(row["relative_path"])
        array = _valid_classifier_array(path)
        if array is None or sha256_array(array) != str(row["array_sha256"]).upper():
            raise ValueError(f"pool array/hash mismatch: {path}")
        if str(row["generator"]) != model:
            raise ValueError(f"pool generator mismatch: {manifest}")
        observed_hashes.append(str(row["array_sha256"]).upper())
    if len(set(observed_hashes)) != len(observed_hashes):
        raise ValueError(f"pool contains duplicate arrays: {manifest}")
    return frame, root


def audit_pools(
    project_root: Path,
    pool_root: Path,
    expected_samples: int = 200,
    seeds: Sequence[int] = EVALUATION_SEEDS,
) -> dict[str, Any]:
    """Strictly audit all three seed pools without loading checkpoints."""
    result: dict[str, Any] = {"schema_version": 1, "seeds": list(seeds), "models": {}}
    for model in ("vae_v3", "diffusion"):
        model_result: dict[str, Any] = {"pools": []}
        for seed in seeds:
            if seed == 42:
                root = project_root / "artifacts/generated_spectrograms"
                manifest = root / model / "manifest.csv"
                if not manifest.is_file():
                    raise FileNotFoundError(f"seed-42 pool manifest missing: {manifest}")
                frame = pd.read_csv(manifest)
                pool_dir = root
                # The historic manifests reference paths below the model root.
                frame = frame.copy()
                frame["relative_path"] = frame["relative_path"].astype(str).str.replace("\\", "/", regex=False)
                frame["relative_path"] = frame["relative_path"].str.replace(f"{model}/", "", regex=False)
                frame["relative_path"] = frame["relative_path"].str.replace("classifier_input/", "classifier_input/", regex=False)
                frame_root = root / model
                metadata_path = frame_root / "generation.json"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if str(metadata.get("checkpoint_sha256", "")).upper() != EXPECTED_CHECKPOINT_SHA256[model]:
                        raise ValueError(f"historic pool checkpoint provenance mismatch: {metadata_path}")
                if len(frame) != expected_samples * len(GENERATOR_CLASSES):
                    raise ValueError(f"historic seed-42 pool is incomplete: {manifest}")
                observed_hashes: list[str] = []
                for row in frame.to_dict(orient="records"):
                    path = frame_root / str(row["relative_path"])
                    array = _valid_classifier_array(path)
                    if array is None or sha256_array(array) != str(row["array_sha256"]).upper():
                        raise ValueError(f"historic seed-42 pool array/hash mismatch: {path}")
                    observed_hashes.append(str(row["array_sha256"]).upper())
                if len(set(observed_hashes)) != len(observed_hashes):
                    raise ValueError(f"historic seed-42 pool contains duplicate arrays: {manifest}")
                count = len(frame)
            else:
                frame, frame_root = _pool_rows(pool_root, model, seed, expected_samples)
                count = len(frame)
            model_result["pools"].append({"seed": int(seed), "sample_count": count})
        result["models"][model] = model_result
    return result


def _load_historic_pool(project_root: Path, model: str, seed: int) -> tuple[pd.DataFrame, Path]:
    if seed != 42:
        raise ValueError("historic helper only handles seed 42")
    root = project_root / "artifacts/generated_spectrograms" / model
    frame = pd.read_csv(root / "manifest.csv")
    frame = frame.copy()
    frame["relative_path"] = frame["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    # Existing manifest paths include the model directory, while root is the
    # model-specific directory used by PoolDataset.
    frame["relative_path"] = frame["relative_path"].str.replace(f"{model}/", "", regex=False)
    return frame, root


def _train_hashes(dataset: ManifestDataset) -> set[str]:
    if dataset.spectrogram_paths is None:
        raise ValueError("copy-risk audit requires a canonical spectrogram cache")
    return {sha256_array(load_cache_array(path, dataset.config)) for path in dataset.spectrogram_paths}


def _copy_risk(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    generated_features: np.ndarray,
    generated_labels: np.ndarray,
) -> dict[str, Any]:
    thresholds: dict[int, float] = {}
    generated_distances: dict[int, np.ndarray] = {}
    for species_index in range(len(CONTENT_SAFE_CLASSES)):
        train = train_features[train_labels == species_index]
        validation = validation_features[validation_labels == species_index]
        generated = generated_features[generated_labels == species_index]
        validation_to_train = pairwise_distances(validation, train).min(axis=1)
        threshold = float(np.quantile(validation_to_train, COPY_RISK_QUANTILE))
        thresholds[species_index] = threshold
        generated_distances[species_index] = pairwise_distances(generated, train).min(axis=1)
    rows = []
    for species_index, species in enumerate(CONTENT_SAFE_CLASSES):
        distances = generated_distances[species_index]
        threshold = thresholds[species_index]
        rows.append(
            {
                "species": species,
                "validation_threshold_quantile": COPY_RISK_QUANTILE,
                "copy_risk_threshold": threshold,
                "generated_count": len(distances),
                "nearest_train_min": float(distances.min()),
                "nearest_train_median": float(np.median(distances)),
                "below_threshold_count": int((distances <= threshold).sum()),
                "below_threshold_fraction": float((distances <= threshold).mean()),
            }
        )
    return {"thresholds": {CONTENT_SAFE_CLASSES[k]: value for k, value in thresholds.items()}, "rows": rows}


def evaluate(
    project_root: Path,
    pool_root: Path,
    report_dir: Path,
    crnn_checkpoint: Path,
    residual_checkpoint: Path,
    test_manifest: Path,
    validation_manifest: Path,
    train_manifest: Path,
    cache_root: Path,
    device: torch.device,
    vae_checkpoint: Path | None = None,
    diffusion_checkpoint: Path | None = None,
    batch_size: int = 128,
    seeds: Sequence[int] = EVALUATION_SEEDS,
    package: bool = True,
) -> dict[str, Any]:
    """Run the frozen evaluator matrix and write the bounded evidence package."""
    project_root = project_root.resolve()
    pool_root = pool_root.resolve()
    report_dir = report_dir.resolve()
    cache_root = resolve_spectrogram_cache_root(project_root, cache_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "classifier_scores").mkdir(parents=True, exist_ok=True)
    (report_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    audit = audit_pools(project_root, pool_root, seeds=seeds)
    config = SpectrogramConfig.from_json((project_root / "configs/spectrogram.json").resolve())
    evaluators = [
        load_evaluator(crnn_checkpoint, device, "crnn"),
        load_evaluator(residual_checkpoint, device, "residual"),
    ]
    train_datasets = {
        evaluator.name: ManifestDataset(train_manifest, None, evaluator.classes, evaluator.config, spectrogram_cache_root=cache_root)
        for evaluator in evaluators
    }
    validation_datasets = {
        evaluator.name: ManifestDataset(validation_manifest, None, evaluator.classes, evaluator.config, spectrogram_cache_root=cache_root)
        for evaluator in evaluators
    }
    test_datasets = {
        evaluator.name: ManifestDataset(test_manifest, None, evaluator.classes, evaluator.config, spectrogram_cache_root=cache_root)
        for evaluator in evaluators
    }
    calibration: list[dict[str, Any]] = []
    real_outputs: dict[str, dict[str, Any]] = {}
    projections: dict[str, tuple[StandardScaler, PCA]] = {}
    for evaluator in evaluators:
        train_outputs = _collect_outputs(evaluator, train_datasets[evaluator.name], device, batch_size)
        validation_outputs = _collect_outputs(evaluator, validation_datasets[evaluator.name], device, batch_size)
        test_outputs = _collect_outputs(evaluator, test_datasets[evaluator.name], device, batch_size, include_arrays=True)
        expected = PRIMARY_CALIBRATION if evaluator.name == "crnn" else RESIDUAL_CALIBRATION
        test_metrics = _classification_metrics(test_outputs, evaluator.classes)
        if abs(test_metrics["accuracy"] - expected["accuracy"]) > 1e-6 or abs(test_metrics["macro_f1"] - expected["macro_f1"]) > 1e-6:
            raise ValueError(f"{evaluator.name} calibration mismatch: {test_metrics['accuracy']:.8f}/{test_metrics['macro_f1']:.8f}")
        calibration.append({"classifier": evaluator.name, "split": "test", **{key: value for key, value in test_metrics.items() if key not in {"predictions", "confusion_matrix", "per_class_recall", "per_class_f1"}}, "expected_accuracy": expected["accuracy"], "expected_macro_f1": expected["macro_f1"]})
        _atomic_json(test_metrics, report_dir / "classifier_scores" / f"{evaluator.name}_test.json")
        pd.DataFrame(test_metrics["confusion_matrix"], index=evaluator.classes, columns=evaluator.classes).to_csv(report_dir / "confusion_matrices" / f"{evaluator.name}_real_test.csv")
        real_outputs[evaluator.name] = {
            "train": train_outputs,
            "validation": validation_outputs,
            "test": test_outputs,
            "test_metrics": test_metrics,
            "train_hashes": _train_hashes(train_datasets[evaluator.name]),
        }
        projections[evaluator.name] = _fit_feature_projection(train_outputs["features"])

    metric_rows: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    for evaluator in evaluators:
        real = real_outputs[evaluator.name]
        scaler, pca = projections[evaluator.name]
        train_z = _project(real["train"]["features"], scaler, pca)
        validation_z = _project(real["validation"]["features"], scaler, pca)
        test_z = _project(real["test"]["features"], scaler, pca)
        for model in ("vae_v3", "diffusion"):
            for seed in seeds:
                if seed == 42:
                    pool_rows, pool_dir = _load_historic_pool(project_root, model, seed)
                else:
                    pool_rows, pool_dir = _pool_rows(pool_root, model, seed)
                pool_dataset = PoolDataset(pool_rows, pool_dir, evaluator.classes)
                generated = _collect_outputs(evaluator, pool_dataset, device, batch_size, include_arrays=True)
                generated_labels = generated["labels"]
                generated_z = _project(generated["features"], scaler, pca)
                generated_metrics = _classification_metrics(generated, evaluator.classes)
                classifier_rows.append({"model": model, "seed": seed, "classifier": evaluator.name, **{key: value for key, value in generated_metrics.items() if key not in {"predictions", "confusion_matrix", "per_class_recall", "per_class_f1"}}})
                pd.DataFrame(generated_metrics["confusion_matrix"], index=evaluator.classes, columns=evaluator.classes).to_csv(report_dir / "confusion_matrices" / f"{model}_seed_{seed}_{evaluator.name}.csv")
                copy_risk = _copy_risk(train_z, real["train"]["labels"], validation_z, real["validation"]["labels"], generated_z, generated_labels)
                generated_hashes = {sha256_array(array) for array in generated["arrays"]}
                exact_train_duplicates = len(generated_hashes & real["train_hashes"])
                for species_index, species in enumerate(CONTENT_SAFE_CLASSES):
                    real_mask = real["test"]["labels"] == species_index
                    generated_mask = generated_labels == evaluator.classes.index(species)
                    if not real_mask.any() or not generated_mask.any():
                        raise ValueError(f"empty species in evaluation: {species}")
                    feature_metrics = _resampled_feature_metrics(test_z[real_mask], generated_z[generated_mask], seed, species_index)
                    image_metrics = _image_metrics(real["test"]["arrays"][real_mask], generated["arrays"][generated_mask])
                    predictions = generated_metrics["predictions"]
                    species_accuracy = float((predictions[generated_mask] == species_index).mean())
                    species_recall = float(generated_metrics["per_class_recall"][species_index])
                    species_f1 = float(generated_metrics["per_class_f1"][species_index])
                    row = {
                        "model": model,
                        "seed": seed,
                        "classifier": evaluator.name,
                        "species": species,
                        "target_label_accuracy": species_accuracy,
                        "species_recall": species_recall,
                        "species_f1": species_f1,
                        "macro_f1": generated_metrics["macro_f1"],
                        "mean_confidence": generated_metrics["mean_confidence"],
                        "mean_entropy": generated_metrics["mean_entropy"],
                        "exact_train_duplicate_count": exact_train_duplicates,
                        "pool_unique_count": len(generated_hashes),
                        **feature_metrics,
                        **image_metrics,
                    }
                    metric_rows.append(row)
                    copy_row = next(item for item in copy_risk["rows"] if item["species"] == species)
                    nearest_rows.append({"model": model, "seed": seed, "classifier": evaluator.name, "exact_train_duplicate_count": exact_train_duplicates, **copy_row})

    metrics_frame = pd.DataFrame(metric_rows)
    nearest_frame = pd.DataFrame(nearest_rows)
    classifier_frame = pd.DataFrame(classifier_rows)
    aggregate_columns = [column for column in metrics_frame.columns if column not in {"model", "classifier", "species"} and column != "seed"]
    aggregate_parts = []
    for keys, group in metrics_frame.groupby(["model", "classifier", "species"], sort=True):
        row = {"model": keys[0], "classifier": keys[1], "species": keys[2]}
        for column in aggregate_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_std"] = float(np.nanstd(values, ddof=1))
            row[f"{column}_min"] = float(np.nanmin(values))
            row[f"{column}_max"] = float(np.nanmax(values))
        aggregate_parts.append(row)
    aggregate_frame = pd.DataFrame(aggregate_parts)
    _atomic_csv(metrics_frame, report_dir / "metrics_per_seed.csv")
    _atomic_csv(aggregate_frame, report_dir / "metrics_aggregate.csv")
    _atomic_csv(nearest_frame, report_dir / "nearest_neighbor_summary.csv")
    _atomic_csv(classifier_frame, report_dir / "classifier_scores.csv")
    _atomic_json(audit, report_dir / "pool_audit.json")

    protocol = {
        "schema_version": 1,
        "title": "Three-seed checkpoint generator evaluation",
        "models": ["vae_v3", "diffusion"],
        "seeds": list(seeds),
        "samples_per_species": 200,
        "total_samples_per_model": 1800,
        "class_order_generator": list(GENERATOR_CLASSES),
        "class_order_classifier": list(CONTENT_SAFE_CLASSES),
        "primary_classifier": "crnn",
        "sensitivity_classifier": "residual",
        "test_manifest": _relative_or_absolute(test_manifest, project_root),
        "train_manifest": _relative_or_absolute(train_manifest, project_root),
        "validation_manifest": _relative_or_absolute(validation_manifest, project_root),
        "cache": _relative_or_absolute(cache_root, project_root),
        "feature_projection": {"fit_split": "content_safe_v2 train", "standardization": "StandardScaler", "pca_components": PCA_COMPONENTS},
        "sample_sensitive_metrics": {"resamples": RESAMPLES, "sample_size_per_species": SAMPLE_SIZE},
        "copy_risk": {"threshold_source": "content_safe_v2 validation nearest-train distances", "quantile": COPY_RISK_QUANTILE},
        "claim_boundary": ["classifier compatibility", "conditioning", "classifier-feature distribution similarity", "diversity", "coverage", "copying risk", "generation-seed stability"],
        "unsupported_claims": ["waveform quality", "human-perceived realism", "native generator test loss", "training stability", "causal augmentation improvement"],
    }
    _atomic_json(protocol, report_dir / "protocol.json")
    provenance = {
        "schema_version": 1,
        "checkpoints": {
            "vae_v3": {"path": "artifacts/models/vae/conditional_vae_v3/conditional_vae_v3_best.pt", "sha256": EXPECTED_CHECKPOINT_SHA256["vae_v3"], "parameter_count": 5_365_025},
            "diffusion": {"path": str(diffusion_checkpoint) if diffusion_checkpoint else "external Desktop checkpoint supplied by user", "sha256": EXPECTED_CHECKPOINT_SHA256["diffusion"], "parameter_count": 18_443_841, "ema_state_dict": True, "note": "external Desktop checkpoint; not copied or tracked"},
        },
        "evaluators": [{"classifier": item.name, "path": _relative_or_absolute(item.checkpoint_path, project_root), "sha256": sha256_file(item.checkpoint_path), "classes": list(item.classes)} for item in evaluators],
        "normalization": {"mean": -51.5400096102764, "std": 14.894513218933453, "conversion": "standardized to dB, per-sample max subtraction, [-80,0] clip, [-1,1]"},
    }
    _atomic_json(provenance, report_dir / "provenance.json")
    summary = {
        "schema_version": 1,
        "title": "Three-seed checkpoint generator evaluation",
        "seed_count": len(seeds),
        "generated_arrays_total": len(seeds) * 2 * len(GENERATOR_CLASSES) * 200,
        "calibration": calibration,
        "primary_classifier": "crnn",
        "sensitivity_classifier": "residual",
        "selection": "No composite score or winner; report encoder-dependent rankings and three-seed mean/std/range.",
        "metric_files": ["metrics_per_seed.csv", "metrics_aggregate.csv", "classifier_scores.csv", "nearest_neighbor_summary.csv"],
    }
    _atomic_json(summary, report_dir / "summary.json")
    return summary


def build_report_charts(report_dir: Path) -> None:
    metrics = pd.read_csv(report_dir / "metrics_per_seed.csv")
    chart_dir = report_dir / "figures"
    chart_dir.mkdir(parents=True, exist_ok=True)
    grouped = metrics.groupby(["model", "classifier", "seed"], as_index=False)["target_label_accuracy"].mean()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for (model, classifier), group in grouped.groupby(["model", "classifier"]):
        axis.plot(group["seed"].astype(str), group["target_label_accuracy"], marker="o", label=f"{model} / {classifier}")
    axis.set(xlabel="Generation seed", ylabel="Mean target-label accuracy", ylim=(0, 1), title="Classifier-view conditioning across generation seeds")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(chart_dir / "conditioning_by_seed.png", dpi=160)
    plt.close(figure)

    grouped = metrics.groupby(["model", "classifier"], as_index=False)["frechet_mean"].mean()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels = [f"{row.model}\n{row.classifier}" for row in grouped.itertuples()]
    axis.bar(labels, grouped["frechet_mean"])
    axis.set_ylabel("Class-conditional feature distance (mean over seeds)")
    axis.set_title("Feature-space distance; lower is closer to real test features")
    figure.tight_layout()
    figure.savefig(chart_dir / "feature_distance_summary.png", dpi=160)
    plt.close(figure)


def write_checksums(report_dir: Path) -> None:
    lines = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            lines.append(f"{digest}  {path.relative_to(report_dir).as_posix()}")
    (report_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
