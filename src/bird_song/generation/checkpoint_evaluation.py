"""Frozen-checkpoint generator evaluation and bounded report packaging.

The metrics in this module are deliberately classifier-view diagnostics.  They
measure conditioning compatibility, feature-space similarity, diversity,
coverage, copy risk, and seed stability; they do not make waveform or human
realism claims.
"""

from __future__ import annotations

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
    GENERATOR_CLASSES,
    _valid_classifier_array,
    deterministic_sample_seed,
    species_slug,
)
from bird_song.generator_safe_validation import load_generator_safe_validation_identity
from bird_song.generation.checkpoint_models import (
    DIFFUSION_CLAMP,
    DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
    DIFFUSION_CHECKPOINT_EPOCH,
    DIFFUSION_DDIM_ETA,
    DIFFUSION_DDIM_STEPS,
    DIFFUSION_GUIDANCE,
    DIFFUSION_TIMESTEPS,
    DIFFUSION_STORED_SAMPLER,
    VAE_REPARAMETERIZATION,
    VAE_TEMPERATURE,
)
from bird_song.generation.posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    POSTERIOR_BANK_EXPECTED_COUNTS,
    POSTERIOR_BANK_SOURCE_MANIFEST,
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


def load_evaluator(path: Path, device: torch.device) -> FrozenEvaluator:
    model, classes, config, checkpoint = load_checkpoint(path.resolve(), device)
    model.eval()
    model.requires_grad_(False)
    if tuple(classes) != CONTENT_SAFE_CLASSES:
        raise ValueError(f"CRNN class order mismatch: {classes}")
    architecture = checkpoint.get("architecture", checkpoint.get("model_config", {}).get("architecture"))
    if architecture != "crnn":
        raise ValueError("generator evaluator is not the selected CRNN checkpoint")
    return FrozenEvaluator("crnn", path.resolve(), model, tuple(classes), config, dict(checkpoint))


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
    if not metadata_path.is_file():
        raise FileNotFoundError(f"pool generation metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        int(metadata.get("schema_version", -1)) != 3
        or str(metadata.get("generator")) != model
        or int(metadata.get("seed", -1)) != int(seed)
        or int(metadata.get("samples_per_class", -1)) != expected_samples
        or list(metadata.get("classes", [])) != list(GENERATOR_CLASSES)
        or int(metadata.get("generation_batch_size", -1)) != 8
    ):
        raise ValueError(f"pool generation settings mismatch: {metadata_path}")
    if model == "vae_v3":
        if (
            abs(float(metadata.get("temperature", -1.0)) - VAE_TEMPERATURE) > 1e-8
            or str(metadata.get("reparameterization", "")) != VAE_REPARAMETERIZATION
            or metadata.get("posterior_bank_contract") != POSTERIOR_BANK_CONTRACT
            or metadata.get("posterior_bank_source_manifest")
            != POSTERIOR_BANK_SOURCE_MANIFEST
            or metadata.get("posterior_bank_derivation")
            != "filtered_existing_posterior_bank"
            or metadata.get("posterior_bank_counts") != POSTERIOR_BANK_EXPECTED_COUNTS
            or metadata.get("vae_checkpoint_retrained") is not False
        ):
            raise ValueError(f"VAE sampling contract mismatch: {metadata_path}")
    elif (
        str(metadata.get("sampler", "")).lower() != "ddim"
        or int(metadata.get("ddim_steps", -1)) != DIFFUSION_DDIM_STEPS
        or abs(float(metadata.get("ddim_eta", -1.0)) - DIFFUSION_DDIM_ETA) > 1e-8
        or abs(float(metadata.get("guidance_weight", -1.0)) - DIFFUSION_GUIDANCE) > 1e-8
        or abs(float(metadata.get("clamp_samples", -1.0)) - DIFFUSION_CLAMP) > 1e-8
        or metadata.get("ema_state_dict") is not True
        or int(metadata.get("checkpoint_epoch", -1)) != DIFFUSION_CHECKPOINT_EPOCH
        or not math.isclose(
            float(metadata.get("checkpoint_best_validation_loss", float("nan"))),
            DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or metadata.get("checkpoint_selection") != "validation_best"
        or metadata.get("stored_sampler_overridden") != DIFFUSION_STORED_SAMPLER
    ):
        raise ValueError(f"diffusion sampling contract mismatch: {metadata_path}")
    frame = pd.read_csv(manifest)
    required = {"species", "relative_path", "generator", "pool_rank", "sample_seed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pool manifest missing columns: {sorted(missing)}")
    if len(frame) != expected_samples * len(GENERATOR_CLASSES):
        raise ValueError(f"pool {manifest} is incomplete: {len(frame)} rows")
    counts = frame["species"].value_counts().to_dict()
    if any(int(counts.get(species, 0)) != expected_samples for species in GENERATOR_CLASSES):
        raise ValueError(f"pool {manifest} is not balanced by species")
    label_by_species = {species: label_id for label_id, species in enumerate(GENERATOR_CLASSES)}
    for species, group in frame.groupby("species", sort=False):
        if str(species) not in label_by_species:
            raise ValueError(f"pool {manifest} contains an unknown species: {species}")
        try:
            ranks = pd.to_numeric(group["pool_rank"], errors="raise").to_numpy(dtype=float)
            sample_seeds = pd.to_numeric(
                group["sample_seed"], errors="raise"
            ).to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"pool {manifest} has non-numeric rank/seed fields") from error
        if (
            not np.isfinite(ranks).all()
            or not np.equal(ranks, np.floor(ranks)).all()
            or sorted(ranks.astype(np.int64).tolist()) != list(range(expected_samples))
            or not np.isfinite(sample_seeds).all()
            or not np.equal(sample_seeds, np.floor(sample_seeds)).all()
        ):
            raise ValueError(f"pool {manifest} has non-canonical ranks or sample seeds")
        expected_seeds = np.asarray(
            [
                deterministic_sample_seed(seed, label_by_species[str(species)], int(rank))
                for rank in ranks
            ],
            dtype=np.int64,
        )
        if not np.array_equal(sample_seeds.astype(np.int64), expected_seeds):
            raise ValueError(f"pool {manifest} has non-canonical sample seeds")

    posterior_inventory: dict[str, dict[str, Any]] = {}
    if model == "vae_v3":
        raw_inventory = metadata.get("posterior_bank_inventory")
        if not isinstance(raw_inventory, dict):
            raise ValueError(f"VAE posterior-bank inventory is missing: {metadata_path}")
        for species in GENERATOR_CLASSES:
            entry = raw_inventory.get(species)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"VAE posterior-bank inventory is missing {species}: {metadata_path}"
                )
            source_indices = entry.get("source_indices")
            relative_wav_paths = entry.get("relative_wav_paths")
            expected_count = POSTERIOR_BANK_EXPECTED_COUNTS[species]
            try:
                recorded_count = int(entry.get("count", -1))
            except (TypeError, ValueError, OverflowError):
                recorded_count = -1
            if (
                recorded_count != expected_count
                or not isinstance(source_indices, list)
                or not isinstance(relative_wav_paths, list)
                or len(source_indices) != expected_count
                or len(relative_wav_paths) != expected_count
            ):
                raise ValueError(
                    f"VAE posterior-bank inventory count mismatch for {species}: {metadata_path}"
                )
            try:
                numeric_indices = [float(value) for value in source_indices]
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"VAE posterior-bank inventory indices are invalid for {species}: "
                    f"{metadata_path}"
                ) from error
            if any(
                not math.isfinite(value) or not value.is_integer()
                for value in numeric_indices
            ):
                raise ValueError(
                    f"VAE posterior-bank inventory indices are invalid for {species}: "
                    f"{metadata_path}"
                )
            normalized_indices = [int(value) for value in numeric_indices]
            normalized_paths = [str(value).replace("\\", "/") for value in relative_wav_paths]
            if (
                len(set(normalized_indices)) != expected_count
                or any(index < 0 for index in normalized_indices)
                or len(set(normalized_paths)) != expected_count
                or any(not path.startswith("wavfiles/") for path in normalized_paths)
            ):
                raise ValueError(
                    f"VAE posterior-bank inventory provenance mismatch for {species}: "
                    f"{metadata_path}"
                )
            posterior_inventory[species] = {
                "source_indices": normalized_indices,
                "relative_wav_paths": normalized_paths,
            }

    observed_arrays: list[bytes] = []
    for row in frame.to_dict(orient="records"):
        path = root / str(row["relative_path"])
        array = _valid_classifier_array(path)
        if array is None:
            raise ValueError(f"invalid pool array: {path}")
        if str(row["generator"]) != model:
            raise ValueError(f"pool generator mismatch: {manifest}")
        if model == "vae_v3":
            species = str(row["species"])
            try:
                temperature = float(row.get("vae_temperature", float("nan")))
                class_count = float(row.get("vae_bank_class_count", float("nan")))
                anchor_index = float(row.get("vae_anchor_index", float("nan")))
                source_index = float(row.get("vae_anchor_source_index", float("nan")))
            except (TypeError, ValueError, OverflowError):
                temperature = class_count = anchor_index = source_index = float("nan")
            inventory = posterior_inventory[species]
            anchor_index_valid = (
                math.isfinite(anchor_index)
                and anchor_index.is_integer()
                and 0 <= int(anchor_index) < POSTERIOR_BANK_EXPECTED_COUNTS[species]
            )
            if (
                not math.isclose(
                    temperature,
                    VAE_TEMPERATURE,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                or str(row.get("vae_reparameterization", "")) != VAE_REPARAMETERIZATION
                or str(row.get("vae_posterior_bank_contract", ""))
                != POSTERIOR_BANK_CONTRACT
                or not math.isfinite(class_count)
                or not class_count.is_integer()
                or int(class_count) != POSTERIOR_BANK_EXPECTED_COUNTS[species]
                or not anchor_index_valid
                or not math.isfinite(source_index)
                or not source_index.is_integer()
                or source_index < 0
                or not str(row.get("vae_anchor_relative_wav_path", "")).startswith(
                    "wavfiles/"
                )
            ):
                raise ValueError(f"VAE anchor provenance mismatch: {manifest}")
            anchor_position = int(anchor_index)
            if (
                int(source_index) != inventory["source_indices"][anchor_position]
                or str(row["vae_anchor_relative_wav_path"]).replace("\\", "/")
                != inventory["relative_wav_paths"][anchor_position]
            ):
                raise ValueError(f"VAE anchor inventory mismatch: {manifest}")
        else:
            try:
                diffusion_row_matches = (
                    str(row.get("sampler", "")).lower() == "ddim"
                    and int(row.get("ddim_steps", -1)) == DIFFUSION_DDIM_STEPS
                    and math.isclose(
                        float(row.get("ddim_eta", float("nan"))),
                        DIFFUSION_DDIM_ETA,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                    and math.isclose(
                        float(row.get("guidance_weight", float("nan"))),
                        DIFFUSION_GUIDANCE,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                    and math.isclose(
                        float(row.get("clamp_samples", float("nan"))),
                        DIFFUSION_CLAMP,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                )
            except (TypeError, ValueError, OverflowError):
                diffusion_row_matches = False
            if not diffusion_row_matches:
                raise ValueError(f"diffusion manifest sampling contract mismatch: {manifest}")
        observed_arrays.append(array.tobytes())
    if len(set(observed_arrays)) != len(observed_arrays):
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
            frame, _ = _pool_rows(pool_root, model, seed, expected_samples)
            count = len(frame)
            model_result["pools"].append({"seed": int(seed), "sample_count": count})
        result["models"][model] = model_result
    return result


def _train_array_bytes(dataset: ManifestDataset) -> set[bytes]:
    if dataset.spectrogram_paths is None:
        raise ValueError("copy-risk audit requires a canonical spectrogram cache")
    return {load_cache_array(path, dataset.config).tobytes() for path in dataset.spectrogram_paths}


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
    validation_protocol_identity = load_generator_safe_validation_identity(
        validation_manifest,
        project_root,
    )
    evaluators = [load_evaluator(crnn_checkpoint, device)]
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
        expected = PRIMARY_CALIBRATION
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
            "train_array_bytes": _train_array_bytes(train_datasets[evaluator.name]),
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
                pool_rows, pool_dir = _pool_rows(pool_root, model, seed)
                pool_dataset = PoolDataset(pool_rows, pool_dir, evaluator.classes)
                generated = _collect_outputs(evaluator, pool_dataset, device, batch_size, include_arrays=True)
                generated_labels = generated["labels"]
                generated_z = _project(generated["features"], scaler, pca)
                generated_metrics = _classification_metrics(generated, evaluator.classes)
                classifier_rows.append({"model": model, "seed": seed, "classifier": evaluator.name, **{key: value for key, value in generated_metrics.items() if key not in {"predictions", "confusion_matrix", "per_class_recall", "per_class_f1"}}})
                pd.DataFrame(generated_metrics["confusion_matrix"], index=evaluator.classes, columns=evaluator.classes).to_csv(report_dir / "confusion_matrices" / f"{model}_seed_{seed}_{evaluator.name}.csv")
                copy_risk = _copy_risk(train_z, real["train"]["labels"], validation_z, real["validation"]["labels"], generated_z, generated_labels)
                generated_arrays = {array.tobytes() for array in generated["arrays"]}
                exact_train_duplicates = len(generated_arrays & real["train_array_bytes"])
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
                        "pool_unique_count": len(generated_arrays),
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
        "schema_version": 3,
        "title": "Three-seed checkpoint generator evaluation",
        "models": ["vae_v3", "diffusion"],
        "seeds": list(seeds),
        "samples_per_species": 200,
        "total_samples_per_model": 1800,
        "class_order_generator": list(GENERATOR_CLASSES),
        "class_order_classifier": list(CONTENT_SAFE_CLASSES),
        "classifier": "crnn",
        "generation_contracts": {
            "vae_v3": {
                "sampling_type": "per_species_posterior_anchor_mixture",
                "temperature": VAE_TEMPERATURE,
                "reparameterization": VAE_REPARAMETERIZATION,
                "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                "posterior_bank_source_manifest": POSTERIOR_BANK_SOURCE_MANIFEST,
                "posterior_bank_counts": POSTERIOR_BANK_EXPECTED_COUNTS,
                "posterior_bank_derivation": "filtered_existing_posterior_bank",
                "vae_checkpoint_retrained": False,
                "pool_reused_after_vae_bank_filter": False,
                "spectrograms_regenerated_after_vae_bank_filter": True,
                "generation_batch_size": 8,
            },
            "diffusion": {
                "sampler": "ddim",
                "timesteps": DIFFUSION_TIMESTEPS,
                "ddim_steps": DIFFUSION_DDIM_STEPS,
                "ddim_eta": DIFFUSION_DDIM_ETA,
                "guidance_weight": DIFFUSION_GUIDANCE,
                "clamp_samples": DIFFUSION_CLAMP,
                "ema_state_dict": True,
                "checkpoint_epoch": DIFFUSION_CHECKPOINT_EPOCH,
                "checkpoint_best_validation_loss": DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
                "checkpoint_selection": "validation_best",
                "stored_sampler_overridden": DIFFUSION_STORED_SAMPLER,
                "pool_reused_after_vae_bank_filter": True,
                "spectrograms_regenerated_after_vae_bank_filter": False,
                "generation_batch_size": 8,
            },
        },
        "test_manifest": _relative_or_absolute(test_manifest, project_root),
        "train_manifest": _relative_or_absolute(train_manifest, project_root),
        "validation_manifest": _relative_or_absolute(validation_manifest, project_root),
        "validation_protocol": validation_protocol_identity,
        "cache": _relative_or_absolute(cache_root, project_root),
        "feature_projection": {"fit_split": "content_safe_v2 train", "standardization": "StandardScaler", "pca_components": PCA_COMPONENTS},
        "sample_sensitive_metrics": {"resamples": RESAMPLES, "sample_size_per_species": SAMPLE_SIZE},
        "copy_risk": {
            "threshold_source": "generator-safe validation nearest-train distances",
            "threshold_source_rows": validation_protocol_identity["validation_counts"]["after"]["rows"],
            "excluded_exact_historical_train_counterparts": len(
                validation_protocol_identity["excluded_validation_relative_wav_paths"]
            ),
            "quantile": COPY_RISK_QUANTILE,
        },
        "claim_boundary": ["classifier compatibility", "conditioning", "classifier-feature distribution similarity", "diversity", "coverage", "copying risk", "generation-seed stability"],
        "unsupported_claims": ["waveform quality", "human-perceived realism", "native generator test loss", "training stability", "causal augmentation improvement"],
    }
    _atomic_json(protocol, report_dir / "protocol.json")
    provenance = {
        "schema_version": 3,
        "checkpoints": {
            "vae_v3": {
                "path": "artifacts/models/vae/conditional_vae_v3/conditional_vae_v3_best.pt",
                "parameter_count": 5_365_025,
                "checkpoint_retrained_for_refresh": False,
                "posterior_bank": "artifacts/models/vae/conditional_vae_v3/class_conditional_posterior_bank.pt",
                "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                "posterior_bank_counts": POSTERIOR_BANK_EXPECTED_COUNTS,
            },
            "diffusion": {
                "path": str(diffusion_checkpoint) if diffusion_checkpoint else "external Desktop checkpoint supplied by user",
                "parameter_count": 18_443_841,
                "ema_state_dict": True,
                "checkpoint_epoch": DIFFUSION_CHECKPOINT_EPOCH,
                "checkpoint_best_validation_loss": DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
                "checkpoint_selection": "validation_best",
                "stored_sampler_overridden": DIFFUSION_STORED_SAMPLER,
                "pool_reused_after_vae_bank_filter": True,
                "spectrograms_regenerated_after_vae_bank_filter": False,
                "note": "external Desktop checkpoint; not copied or tracked",
            },
        },
        "evaluators": [{"classifier": item.name, "path": _relative_or_absolute(item.checkpoint_path, project_root), "classes": list(item.classes)} for item in evaluators],
        "validation_protocol": validation_protocol_identity,
        "normalization": {"mean": -51.5400096102764, "std": 14.894513218933453, "conversion": "standardized to dB, per-sample max subtraction, [-80,0] clip, [-1,1]"},
    }
    _atomic_json(provenance, report_dir / "provenance.json")
    summary = {
        "schema_version": 3,
        "title": "Three-seed checkpoint generator evaluation",
        "seed_count": len(seeds),
        "generated_arrays_total": len(seeds) * 2 * len(GENERATOR_CLASSES) * 200,
        "calibration": calibration,
        "classifier": "crnn",
        "refresh_actions": {
            "vae_v3": "filtered posterior bank and regenerated all three pools",
            "diffusion": "reused audited DDIM pools; no diffusion spectrogram regeneration",
        },
        "selection": "No composite score or winner; report three-seed mean, sample standard deviation, and range.",
        "metric_files": ["metrics_per_seed.csv", "metrics_aggregate.csv", "classifier_scores.csv", "nearest_neighbor_summary.csv"],
    }
    _atomic_json(summary, report_dir / "summary.json")
    return summary


def build_report_charts(report_dir: Path) -> None:
    metrics = pd.read_csv(report_dir / "metrics_per_seed.csv")
    classifiers = set(metrics["classifier"].astype(str))
    if classifiers != {"crnn"}:
        raise ValueError(f"Expected CRNN-only report metrics, found classifiers: {sorted(classifiers)}")
    chart_dir = report_dir / "figures"
    chart_dir.mkdir(parents=True, exist_ok=True)
    grouped = metrics.groupby(["model", "seed"], as_index=False)["target_label_accuracy"].mean()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for model, group in grouped.groupby("model"):
        axis.plot(group["seed"].astype(str), group["target_label_accuracy"], marker="o", label=model)
    axis.set(xlabel="Generation seed", ylabel="Mean target-label accuracy", ylim=(0, 1), title="Classifier-view conditioning across generation seeds")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(chart_dir / "conditioning_by_seed.png", dpi=160)
    plt.close(figure)

    grouped = metrics.groupby("model", as_index=False)["frechet_mean"].mean()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels = [row.model for row in grouped.itertuples()]
    axis.bar(labels, grouped["frechet_mean"])
    axis.set_ylabel("Class-conditional feature distance (mean over seeds)")
    axis.set_title("Feature-space distance; lower is closer to real test features")
    figure.tight_layout()
    figure.savefig(chart_dir / "feature_distance_summary.png", dpi=160)
    plt.close(figure)
