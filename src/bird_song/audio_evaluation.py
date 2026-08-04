from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

from .data import InferenceDataset
from .runtime import load_checkpoint


def select_balanced_recordings(
    rows: pd.DataFrame,
    per_class: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a class-balanced subset while maximizing unique recording IDs first."""

    if per_class < 1:
        raise ValueError("per_class must be positive")
    required = {"name", "relative_wav_path"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    selected = []
    for class_index, class_name in enumerate(sorted(rows["name"].unique())):
        candidates = rows.loc[rows["name"] == class_name].copy()
        if len(candidates) < per_class:
            raise ValueError(f"Class {class_name!r} has only {len(candidates)} rows, need {per_class}")
        candidates = candidates.sample(frac=1.0, random_state=seed + class_index)
        if "id" in candidates.columns:
            first_per_recording = candidates.drop_duplicates("id", keep="first")
            chosen = first_per_recording.head(per_class)
            if len(chosen) < per_class:
                remaining = candidates.loc[~candidates.index.isin(chosen.index)]
                chosen = pd.concat((chosen, remaining.head(per_class - len(chosen))))
        else:
            chosen = candidates.head(per_class)
        selected.append(chosen)
    return pd.concat(selected).sort_values(["name", "relative_wav_path"]).reset_index(drop=True)


def waveform_diagnostics(waveform: torch.Tensor, expected_samples: int) -> dict[str, float | bool | int]:
    flattened = waveform.detach().float().cpu().flatten()
    finite = bool(torch.isfinite(flattened).all())
    peak = float(flattened.abs().amax()) if flattened.numel() else float("nan")
    clipped_fraction = float((flattened.abs() >= 0.9999).float().mean()) if flattened.numel() else 1.0
    return {
        "num_samples": flattened.numel(),
        "finite": finite,
        "peak": peak,
        "clipped_fraction": clipped_fraction,
        "valid": finite and flattened.numel() == expected_samples and peak <= 1.0001,
    }


def multi_resolution_stft_error(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    resolutions: Iterable[tuple[int, int]] = ((512, 128), (1024, 256), (2048, 512)),
) -> dict[str, float]:
    reference = reference.detach().float().cpu().flatten()
    estimate = estimate.detach().float().cpu().flatten()
    if reference.shape != estimate.shape:
        raise ValueError(f"Waveform shapes differ: {tuple(reference.shape)} vs {tuple(estimate.shape)}")
    spectral_convergence = []
    log_magnitude_l1 = []
    for n_fft, hop_length in resolutions:
        window = torch.hann_window(n_fft)
        reference_stft = torch.stft(
            reference,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        estimate_stft = torch.stft(
            estimate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        spectral_convergence.append(
            float((reference_stft - estimate_stft).norm() / reference_stft.norm().clamp_min(1e-8))
        )
        log_magnitude_l1.append(
            float(
                (
                    torch.log(reference_stft.clamp_min(1e-7))
                    - torch.log(estimate_stft.clamp_min(1e-7))
                )
                .abs()
                .mean()
            )
        )
    return {
        "mrstft_spectral_convergence": float(np.mean(spectral_convergence)),
        "mrstft_log_magnitude_l1": float(np.mean(log_magnitude_l1)),
    }


def _classifier_features(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if all(
        hasattr(model, attribute)
        for attribute in ("stem", "features", "average_pool", "maximum_pool")
    ):
        features = model.features(model.stem(inputs))
        pooled = torch.cat((model.average_pool(features), model.maximum_pool(features)), dim=1)
        return pooled.flatten(1)
    return model(inputs)


@torch.inference_mode()
def evaluate_audio_classifier(
    checkpoint_path: Path,
    paths: list[Path],
    targets: list[str],
    device: torch.device,
    *,
    batch_size: int = 64,
    workers: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    if len(paths) != len(targets) or not paths:
        raise ValueError("paths and targets must be non-empty and equal length")
    model, classes, config, _ = load_checkpoint(checkpoint_path, device)
    class_to_index = {name: index for index, name in enumerate(classes)}
    unknown = sorted(set(targets) - set(classes))
    if unknown:
        raise ValueError(f"Classifier checkpoint does not contain target classes: {unknown}")
    dataset = InferenceDataset(paths, config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model.eval()
    rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    offset = 0
    for spectrograms, batch_paths in loader:
        spectrograms = spectrograms.to(device, non_blocking=True)
        probabilities = model(spectrograms).softmax(1).cpu()
        batch_embeddings = _classifier_features(model, spectrograms).cpu().numpy()
        embeddings.append(batch_embeddings)
        for local_index, (path, probability) in enumerate(zip(batch_paths, probabilities)):
            target = targets[offset + local_index]
            prediction = int(probability.argmax())
            row = {
                "path": path,
                "target": target,
                "prediction": classes[prediction],
                "correct": classes[prediction] == target,
                "confidence": float(probability[prediction]),
            }
            row.update({f"p_{name}": float(probability[index]) for index, name in enumerate(classes)})
            rows.append(row)
        offset += len(batch_paths)
    frame = pd.DataFrame(rows)
    return frame, np.concatenate(embeddings), float(frame["correct"].mean())


def embedding_frechet_distance(reference: np.ndarray, estimate: np.ndarray) -> float:
    """FAD-style Gaussian Fréchet distance for caller-provided audio embeddings."""

    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 2 or estimate.ndim != 2 or reference.shape[1] != estimate.shape[1]:
        raise ValueError("Embedding matrices must be 2-D with equal feature dimensions")
    if len(reference) < 2 or len(estimate) < 2:
        raise ValueError("At least two embeddings per condition are required")
    mean_reference = reference.mean(axis=0)
    mean_estimate = estimate.mean(axis=0)
    covariance_reference = np.atleast_2d(np.cov(reference, rowvar=False))
    covariance_estimate = np.atleast_2d(np.cov(estimate, rowvar=False))
    covariance_mean = sqrtm(covariance_reference @ covariance_estimate)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = mean_reference - mean_estimate
    distance = (
        difference @ difference
        + np.trace(covariance_reference)
        + np.trace(covariance_estimate)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(max(distance, 0.0))


def listening_median(path: Path, condition: str) -> float:
    ratings = pd.read_csv(path)
    required = {"condition", "bird_likeness"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Listening ratings are missing columns: {sorted(missing)}")
    values = pd.to_numeric(
        ratings.loc[ratings["condition"] == condition, "bird_likeness"],
        errors="coerce",
    ).dropna()
    if values.empty:
        raise ValueError(f"No bird-likeness ratings found for condition {condition!r}")
    if not values.between(1, 5).all():
        raise ValueError("bird_likeness ratings must be between 1 and 5")
    return float(values.median())
