from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from bird_song.augmentation.data import GeneratedSpectrogramDataset, audit_generated_pool
from bird_song.classifier.model import build_classifier, count_trainable_parameters
from bird_song.config import DEFAULT_CLASSES, SpectrogramConfig
from bird_song.data import ManifestDataset
from bird_song.runtime import atomic_torch_save, load_checkpoint, save_json, seed_everything
from bird_song.spectrogram_cache import audit_spectrogram_cache, sha256_file


DEFAULT_RATIOS = (50, 100, 200)
DEFAULT_SEEDS = (42, 123, 777)
DEFAULT_STEPS = 1_440
DEFAULT_VALIDATE_EVERY = 36
DEFAULT_BATCH_SIZE = 64
PROTOCOL_VERSION = 2
MODEL_WIDTH = 32
MODEL_DROPOUT = 0.30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 5.0


def _training_protocol(device: torch.device) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": "crnn",
        "model_width": MODEL_WIDTH,
        "model_dropout": MODEL_DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "label_smoothing": LABEL_SMOOTHING,
        "gradient_clip_max_norm": GRAD_CLIP_NORM,
        "amp_enabled": device.type == "cuda",
        "real_input_transform": "precomputed deterministic normalized log-mel; no post-cache augmentation",
        "generated_input_transform": (
            "precomputed normalized log-mel; independent p=0.5 frequency mask up to 12 bins "
            "and p=0.5 time mask up to 16 frames"
        ),
    }


def _run_signature(
    *,
    generator_name: str,
    ratio_per_species: int,
    seed: int,
    steps: int,
    validate_every: int,
    batch_size: int,
    workers: int,
    train_manifest: Path,
    validation_manifest: Path,
    real_cache_root: Path,
    pool_manifest: Path,
    spectrogram_config_path: Path,
    config: SpectrogramConfig,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generator": generator_name,
        "ratio_per_species": int(ratio_per_species),
        "seed": int(seed),
        "steps": int(steps),
        "validate_every": int(validate_every),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "real_cache_manifest_sha256": sha256_file(real_cache_root / "spectrogram_manifest.csv"),
        "pool_manifest_sha256": sha256_file(pool_manifest),
        "spectrogram_config_sha256": sha256_file(spectrogram_config_path),
        "spectrogram_config": config.to_dict(),
        "training_protocol": _training_protocol(device),
    }


def _validate_existing_run(output_dir: Path, expected_signature: dict[str, Any]) -> Path:
    checkpoint_path = output_dir / "best.pt"
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Existing checkpoint has no strict run config: {checkpoint_path}. "
            "Use a fresh --run-root or pass --overwrite."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_signature = checkpoint.get("run_signature")
    config_signature = run_config.get("run_signature")
    if checkpoint_signature != expected_signature or config_signature != expected_signature:
        raise RuntimeError(
            f"Existing run is incompatible with the requested data or protocol: {output_dir}. "
            "Use a fresh --run-root or pass --overwrite."
        )
    return checkpoint_path


def _guard_legacy_overwrite(checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("run_signature") is None:
        raise RuntimeError(
            f"Refusing to overwrite legacy evidence without a strict run signature: {checkpoint_path}. "
            "Use a fresh --run-root."
        )


def _common_run_signature(signature: dict[str, Any]) -> dict[str, Any]:
    arm_specific = {"generator", "ratio_per_species", "seed", "pool_manifest_sha256"}
    return {key: value for key, value in signature.items() if key not in arm_specific}


def _portable(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _loader(
    dataset,
    *,
    batch_size: int,
    workers: int,
    training: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training and len(dataset) >= batch_size,
    )


def _metrics(confusion: np.ndarray) -> tuple[float, float, list[float]]:
    true_positives = np.diag(confusion).astype(float)
    precision = np.divide(
        true_positives,
        confusion.sum(axis=0),
        out=np.zeros_like(true_positives),
        where=confusion.sum(axis=0) > 0,
    )
    recall = np.divide(
        true_positives,
        confusion.sum(axis=1),
        out=np.zeros_like(true_positives),
        where=confusion.sum(axis=1) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=precision + recall > 0,
    )
    return float(true_positives.sum() / max(int(confusion.sum()), 1)), float(f1.mean()), recall.tolist()


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((len(DEFAULT_CLASSES), len(DEFAULT_CLASSES)), dtype=np.int64)
    total_loss = 0.0
    total_items = 0
    for specs, labels, _ in loader:
        specs = specs.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(specs)
        predictions = logits.argmax(1).cpu().numpy()
        truth = labels.numpy()
        confusion += np.bincount(
            truth * len(DEFAULT_CLASSES) + predictions,
            minlength=len(DEFAULT_CLASSES) ** 2,
        ).reshape(confusion.shape)
        total_loss += float(criterion(logits, labels_device)) * labels.numel()
        total_items += labels.numel()
    accuracy, macro_f1, recall = _metrics(confusion)
    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_species_recall": dict(zip(DEFAULT_CLASSES, recall)),
        "confusion": confusion.tolist(),
        "sample_count": int(confusion.sum()),
    }


def build_real_dataset(
    manifest: Path,
    cache_root: Path,
    config: SpectrogramConfig,
    *,
    training: bool,
) -> ManifestDataset:
    return ManifestDataset(
        manifest,
        None,
        DEFAULT_CLASSES,
        config,
        training=training,
        spectrogram_cache_root=cache_root,
    )


def build_mixed_training_dataset(
    train_manifest: Path,
    real_cache_root: Path,
    generated_cache_root: Path,
    pool_manifest: Path,
    config: SpectrogramConfig,
    ratio_per_species: int,
):
    real = build_real_dataset(train_manifest, real_cache_root, config, training=True)
    generated = GeneratedSpectrogramDataset(
        pool_manifest,
        generated_cache_root,
        DEFAULT_CLASSES,
        config,
        ratio_per_species,
        training=True,
    )
    return ConcatDataset((real, generated)), len(real), len(generated)


def train_condition(
    *,
    project_root: Path,
    generator_name: str,
    ratio_per_species: int,
    seed: int,
    train_manifest: Path,
    validation_manifest: Path,
    real_cache_root: Path,
    generated_cache_root: Path,
    pool_manifest: Path,
    spectrogram_config_path: Path,
    output_dir: Path,
    config: SpectrogramConfig,
    device: torch.device,
    steps: int = DEFAULT_STEPS,
    validate_every: int = DEFAULT_VALIDATE_EVERY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 0,
    overwrite: bool = False,
) -> Path:
    checkpoint_path = output_dir / "best.pt"
    signature = _run_signature(
        generator_name=generator_name,
        ratio_per_species=ratio_per_species,
        seed=seed,
        steps=steps,
        validate_every=validate_every,
        batch_size=batch_size,
        workers=workers,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        real_cache_root=real_cache_root,
        pool_manifest=pool_manifest,
        spectrogram_config_path=spectrogram_config_path,
        config=config,
        device=device,
    )
    if checkpoint_path.is_file():
        if not overwrite:
            return _validate_existing_run(output_dir, signature)
        _guard_legacy_overwrite(checkpoint_path)
    if steps < 1 or validate_every < 1 or steps % validate_every:
        raise ValueError("steps and validate_every must be positive, and steps must be divisible by validate_every")
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data, real_count, generated_count = build_mixed_training_dataset(
        train_manifest,
        real_cache_root,
        generated_cache_root,
        pool_manifest,
        config,
        ratio_per_species,
    )
    validation_data = build_real_dataset(validation_manifest, real_cache_root, config, training=False)
    train_loader = _loader(train_data, batch_size=batch_size, workers=workers, training=True, seed=seed)
    validation_loader = _loader(
        validation_data,
        batch_size=batch_size,
        workers=workers,
        training=False,
        seed=seed,
    )
    model = build_classifier(
        "crnn", num_classes=len(DEFAULT_CLASSES), dropout=MODEL_DROPOUT, width=MODEL_WIDTH
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -math.inf
    best_step = 0
    history: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        try:
            specs, labels, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            specs, labels, _ = next(iterator)
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = criterion(model(specs), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step % validate_every:
            continue
        validation = evaluate_model(model, validation_loader, device)
        history.append(
            {
                "step": step,
                "train_loss": float(loss.detach()),
                "validation": validation,
                "seconds": time.perf_counter() - started,
            }
        )
        current_f1 = float(validation["macro_f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_step = step
            atomic_torch_save(
                {
                    "format_version": 3,
                    "model_state": model.state_dict(),
                    "model_config": model.metadata(),
                    "architecture": "crnn",
                    "trainable_parameters": count_trainable_parameters(model),
                    "classes": list(DEFAULT_CLASSES),
                    "spectrogram_config": config.to_dict(),
                    "seed": seed,
                    "generator": generator_name,
                    "ratio": ratio_per_species,
                    "steps": steps,
                    "validation_every": validate_every,
                    "best_step": best_step,
                    "best_validation_macro_f1": best_f1,
                    "validation": validation,
                    "spectrogram_cache": _portable(real_cache_root, project_root),
                    "generated_cache": _portable(generated_cache_root, project_root),
                    "run_signature": signature,
                },
                checkpoint_path,
            )
        if step % (validate_every * 4) == 0:
            print(
                f"generator={generator_name} ratio={ratio_per_species} seed={seed} step={step} "
                f"val_accuracy={float(validation['accuracy']):.4f} val_macro_f1={current_f1:.4f}",
                flush=True,
            )
    save_json(
        output_dir / "config.json",
        {
            "generator": generator_name,
            "ratio_per_species": ratio_per_species,
            "seed": seed,
            "device": str(device),
            "steps": steps,
            "validation_every": validate_every,
            "batch_size": batch_size,
            "workers": workers,
            "real_rows": real_count,
            "generated_rows": generated_count,
            "synthetic_share": generated_count / (real_count + generated_count),
            "real_cache": _portable(real_cache_root, project_root),
            "generated_cache": _portable(generated_cache_root, project_root),
            "pool_manifest": _portable(pool_manifest, project_root),
            "trainable_parameters": count_trainable_parameters(model),
            "elapsed_seconds": time.perf_counter() - started,
            "best_step": best_step,
            "best_validation_macro_f1": best_f1,
            "run_signature": signature,
            "data_provenance": {
                "train_manifest": _portable(train_manifest, project_root),
                "validation_manifest": _portable(validation_manifest, project_root),
                "real_cache_manifest": _portable(real_cache_root / "spectrogram_manifest.csv", project_root),
                "pool_manifest": _portable(pool_manifest, project_root),
                "spectrogram_config": _portable(spectrogram_config_path, project_root),
            },
        },
    )
    save_json(output_dir / "history.json", history)
    return checkpoint_path


def run_sweep(
    *,
    project_root: Path,
    generators: Iterable[str],
    ratios: Iterable[int],
    seeds: Iterable[int],
    run_root: Path,
    train_manifest: Path,
    validation_manifest: Path,
    real_cache_root: Path,
    generated_cache_root: Path,
    spectrogram_config_path: Path,
    config: SpectrogramConfig,
    device: torch.device,
    steps: int,
    validate_every: int,
    batch_size: int,
    workers: int,
    overwrite: bool,
) -> list[Path]:
    generators = tuple(dict.fromkeys(str(value) for value in generators))
    ratios = tuple(dict.fromkeys(int(value) for value in ratios))
    seeds = tuple(dict.fromkeys(int(value) for value in seeds))
    if not generators or not ratios or not seeds:
        raise ValueError("At least one generator, ratio, and seed is required")
    if any(value < 1 for value in ratios):
        raise ValueError("All synthetic ratios must be positive")
    if SpectrogramConfig.from_json(spectrogram_config_path) != config:
        raise ValueError("The supplied spectrogram config object does not match its source JSON")
    audit_spectrogram_cache(real_cache_root, config)
    checkpoints = []
    for generator_name in generators:
        pool_manifest = generated_cache_root / generator_name / "manifest.csv"
        if not pool_manifest.is_file():
            raise FileNotFoundError(f"Generated cache manifest is missing: {pool_manifest}")
        audit_generated_pool(pool_manifest, generated_cache_root, DEFAULT_CLASSES, config)
        for ratio in ratios:
            for seed in seeds:
                output_dir = run_root / generator_name / f"ratio_{ratio}" / f"seed_{seed}"
                print(f"start generator={generator_name} ratio={ratio} seed={seed} device={device}", flush=True)
                checkpoint = train_condition(
                    project_root=project_root,
                    generator_name=generator_name,
                    ratio_per_species=ratio,
                    seed=seed,
                    train_manifest=train_manifest,
                    validation_manifest=validation_manifest,
                    real_cache_root=real_cache_root,
                    generated_cache_root=generated_cache_root,
                    pool_manifest=pool_manifest,
                    spectrogram_config_path=spectrogram_config_path,
                    output_dir=output_dir,
                    config=config,
                    device=device,
                    steps=steps,
                    validate_every=validate_every,
                    batch_size=batch_size,
                    workers=workers,
                    overwrite=overwrite,
                )
                checkpoints.append(checkpoint)
                print(f"complete checkpoint={checkpoint}", flush=True)
    return checkpoints


def select_ratio(means: dict[int, float], tie_tolerance: float = 0.002) -> int:
    if not means:
        raise ValueError("At least one validation mean is required")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    best = max(float(value) for value in means.values())
    return min(int(ratio) for ratio, value in means.items() if best - float(value) <= tie_tolerance)


def _sample_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _pool_provenance(generated_cache_root: Path, generator_name: str, project_root: Path) -> dict[str, Any]:
    directory = generated_cache_root / generator_name
    manifest = directory / "manifest.csv"
    generation = directory / "generation.json"
    return {
        "root": _portable(directory, project_root),
        "manifest_sha256": sha256_file(manifest),
        "generation_sha256": sha256_file(generation),
        "generation": json.loads(generation.read_text(encoding="utf-8")),
        "rows": int(len(pd.read_csv(manifest))),
    }


def _logical_manifest_keys(path: Path) -> list[tuple[str, str, str]]:
    rows = pd.read_csv(path)
    required = ["split", "name", "relative_wav_path"]
    missing = set(required) - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest is missing logical identity columns {sorted(missing)}: {path}")
    if rows.empty or rows[required].isna().any().any():
        raise ValueError(f"Manifest is empty or has missing logical identity values: {path}")
    keys = [tuple(str(value) for value in row) for row in rows[required].itertuples(index=False, name=None)]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Manifest contains duplicate logical rows: {path}")
    return sorted(keys)


def _validate_baseline_test_manifest(
    baseline_payload: dict[str, Any],
    test_manifest: Path,
    project_root: Path,
) -> dict[str, Any]:
    metadata = baseline_payload.get("manifest")
    if not isinstance(metadata, dict):
        raise ValueError("Baseline metrics do not record test-manifest provenance")
    recorded_path_value = metadata.get("path")
    recorded_hash = str(metadata.get("sha256", "")).upper()
    recorded_count = int(metadata.get("sample_count", -1))
    if not recorded_path_value or len(recorded_hash) != 64 or recorded_count < 1:
        raise ValueError("Baseline test-manifest provenance is incomplete")

    recorded_relative = Path(str(recorded_path_value).replace("\\", "/"))
    recorded_path = (
        recorded_relative.resolve()
        if recorded_relative.is_absolute()
        else (project_root.resolve() / recorded_relative).resolve()
    )
    supplied_path = test_manifest.resolve()
    supplied_keys = _logical_manifest_keys(supplied_path)
    if len(supplied_keys) != recorded_count:
        raise ValueError("Supplied test manifest does not match the baseline sample count")
    if recorded_path.is_file():
        if sha256_file(recorded_path) != recorded_hash:
            raise ValueError(f"Recorded baseline test manifest hash is stale: {recorded_path}")
        if _logical_manifest_keys(recorded_path) != supplied_keys:
            raise ValueError("Supplied test manifest has different logical clips from the baseline evaluation")
    elif sha256_file(supplied_path) != recorded_hash:
        raise ValueError(
            "The recorded baseline test manifest is unavailable and the supplied manifest hash differs"
        )
    return metadata


def select_and_evaluate(
    *,
    project_root: Path,
    generators: Iterable[str],
    ratios: Iterable[int],
    seeds: Iterable[int],
    run_root: Path,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    real_cache_root: Path,
    generated_cache_root: Path,
    baseline_metrics: Path,
    spectrogram_config_path: Path,
    output_path: Path,
    config: SpectrogramConfig,
    device: torch.device,
    batch_size: int,
    workers: int,
    overwrite_report: bool = False,
) -> Path:
    if output_path.exists() and not overwrite_report:
        raise FileExistsError(f"Evaluation report already exists: {output_path}; pass --overwrite-report intentionally")
    requested_generators = tuple(dict.fromkeys(str(value) for value in generators))
    requested_ratios = sorted(set(int(value) for value in ratios))
    requested_seeds = sorted(set(int(seed) for seed in seeds))
    if not requested_generators or not requested_ratios or not requested_seeds:
        raise ValueError("At least one generator, ratio, and seed is required")
    _, real_cache_audit = audit_spectrogram_cache(real_cache_root, config)
    train_data = build_real_dataset(train_manifest, real_cache_root, config, training=False)
    validation_data = build_real_dataset(validation_manifest, real_cache_root, config, training=False)
    test_data = build_real_dataset(test_manifest, real_cache_root, config, training=False)
    test_loader = _loader(test_data, batch_size=batch_size, workers=workers, training=False, seed=0)
    baseline_payload = json.loads(baseline_metrics.read_text(encoding="utf-8"))
    baseline_manifest = _validate_baseline_test_manifest(baseline_payload, test_manifest, project_root)
    baseline_report = baseline_payload["report"]
    baseline = {
        "comparison_role": "historical_single_checkpoint",
        "checkpoint": baseline_payload["checkpoint"],
        "manifest": baseline_manifest,
        "accuracy": float(baseline_report["accuracy"]),
        "macro_f1": float(baseline_report["macro avg"]["f1-score"]),
    }
    report: dict[str, Any] = {
        "format_version": 2,
        "protocol": {
            "real_training_rows": len(train_data),
            "validation_rows": len(validation_data),
            "test_rows": len(test_data),
            "ratios_per_species": requested_ratios,
            "seeds": requested_seeds,
            "selection_metric": "mean best validation macro_f1; choose the smallest ratio within 0.002 of the best",
            "ratio_tie_tolerance": 0.002,
            "test_policy": "evaluate only the validation-selected ratio for each generator",
            "real_cache": _portable(real_cache_root, project_root),
            "real_cache_manifest_sha256": sha256_file(real_cache_root / "spectrogram_manifest.csv"),
            "train_manifest": _portable(train_manifest, project_root),
            "train_manifest_sha256": sha256_file(train_manifest),
            "validation_manifest": _portable(validation_manifest, project_root),
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "test_manifest": _portable(test_manifest, project_root),
            "test_manifest_sha256": sha256_file(test_manifest),
            "spectrogram_config": _portable(spectrogram_config_path, project_root),
            "spectrogram_config_sha256": sha256_file(spectrogram_config_path),
            "baseline_metrics": _portable(baseline_metrics, project_root),
            "baseline_metrics_sha256": sha256_file(baseline_metrics),
        },
        "baseline": baseline,
        "pools": {
            generator_name: {
                **_pool_provenance(generated_cache_root, generator_name, project_root),
                "integrity_audit": audit_generated_pool(
                    generated_cache_root / generator_name / "manifest.csv",
                    generated_cache_root,
                    DEFAULT_CLASSES,
                    config,
                ),
            }
            for generator_name in requested_generators
        },
        "validation_selection": {},
        "test": {},
        "caveats": [
            "The baseline is one historical seed-777 checkpoint trained with the earlier waveform path, not a matched cached-real-only three-seed control.",
            "Interpret the selected ratio against seed-to-seed variation; this sweep does not establish a stable optimum beyond its tested ratios and generated pool.",
        ],
    }
    if int(real_cache_audit["cross_split_duplicate_group_count"]) > 0:
        test_involved = any(
            "test" in group["splits"] for group in real_cache_audit["cross_split_duplicate_groups"]
        )
        test_note = (
            "At least one exact-content group involves test."
            if test_involved
            else "No exact-content group involves the historical test split."
        )
        report["caveats"].append(
            "The shared historical cache contains exact-content groups across manifest splits; use content-safe manifests "
            f"for future experiments. {test_note}"
        )
    provenance_levels: set[str] = set()
    maintained_common_signature: dict[str, Any] | None = None
    for generator_name in requested_generators:
        candidates: dict[int, list[tuple[int, Path, dict[str, Any]]]] = {}
        for ratio in requested_ratios:
            records = []
            for seed in requested_seeds:
                checkpoint = run_root / generator_name / f"ratio_{ratio}" / f"seed_{seed}" / "best.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Sweep checkpoint is missing: {checkpoint}")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if payload.get("generator") != generator_name or int(payload.get("ratio", -1)) != ratio:
                    raise ValueError(f"Checkpoint metadata does not match its arm: {checkpoint}")
                if int(payload.get("seed", -1)) != seed:
                    raise ValueError(f"Checkpoint seed does not match its arm: {checkpoint}")
                run_config_path = checkpoint.parent / "config.json"
                if not run_config_path.is_file():
                    raise FileNotFoundError(f"Sweep run config is missing: {run_config_path}")
                run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                signature = payload.get("run_signature")
                if signature is None:
                    if int(run_config.get("real_rows", -1)) != len(train_data):
                        raise ValueError(
                            f"Legacy checkpoint row count does not match the supplied train manifest: {checkpoint}"
                        )
                    provenance_levels.add("legacy_row_count_only")
                else:
                    if run_config.get("run_signature") != signature:
                        raise ValueError(f"Checkpoint and run config signatures disagree: {checkpoint.parent}")
                    expected_arm = {
                        "generator": generator_name,
                        "ratio_per_species": ratio,
                        "seed": seed,
                    }
                    arm_mismatches = [
                        key for key, value in expected_arm.items() if signature.get(key) != value
                    ]
                    if arm_mismatches:
                        raise ValueError(
                            f"Checkpoint run signature does not match its arm ({arm_mismatches}): {checkpoint}"
                        )
                    expected_hashes = {
                        "train_manifest_sha256": sha256_file(train_manifest),
                        "validation_manifest_sha256": sha256_file(validation_manifest),
                        "real_cache_manifest_sha256": sha256_file(
                            real_cache_root / "spectrogram_manifest.csv"
                        ),
                        "pool_manifest_sha256": sha256_file(
                            generated_cache_root / generator_name / "manifest.csv"
                        ),
                        "spectrogram_config_sha256": sha256_file(spectrogram_config_path),
                    }
                    mismatched = [key for key, value in expected_hashes.items() if signature.get(key) != value]
                    if mismatched:
                        raise ValueError(
                            f"Checkpoint provenance does not match supplied inputs ({mismatched}): {checkpoint}"
                        )
                    common_signature = _common_run_signature(signature)
                    if maintained_common_signature is None:
                        maintained_common_signature = common_signature
                    elif maintained_common_signature != common_signature:
                        raise ValueError(
                            "Requested checkpoints use inconsistent maintained training protocols"
                        )
                    provenance_levels.add("strict_hashes")
                records.append((seed, checkpoint, payload))
            candidates[ratio] = records
        means = {
            ratio: float(np.mean([float(payload["best_validation_macro_f1"]) for _, _, payload in records]))
            for ratio, records in candidates.items()
        }
        standard_deviations = {
            ratio: _sample_sd([float(payload["best_validation_macro_f1"]) for _, _, payload in records])
            for ratio, records in candidates.items()
        }
        selected = select_ratio(means)
        report["validation_selection"][generator_name] = {
            "mean_macro_f1_by_ratio": {str(key): value for key, value in means.items()},
            "sample_sd_macro_f1_by_ratio": {str(key): value for key, value in standard_deviations.items()},
            "selected_ratio": selected,
        }
        seed_results = []
        for seed, checkpoint, _ in candidates[selected]:
            model, classes, loaded_config, payload = load_checkpoint(checkpoint, device)
            if classes != DEFAULT_CLASSES or loaded_config != config:
                raise ValueError(f"Checkpoint class/config contract does not match this evaluation: {checkpoint}")
            result = evaluate_model(model, test_loader, device)
            result.update(
                {
                    "seed": seed,
                    "checkpoint": _portable(checkpoint, project_root),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "best_validation_macro_f1": float(payload["best_validation_macro_f1"]),
                }
            )
            seed_results.append(result)
        accuracies = [float(result["accuracy"]) for result in seed_results]
        macro_f1s = [float(result["macro_f1"]) for result in seed_results]
        mean_accuracy = float(np.mean(accuracies))
        mean_macro_f1 = float(np.mean(macro_f1s))
        report["test"][generator_name] = {
            "selected_ratio": selected,
            "seeds": seed_results,
            "mean_accuracy": mean_accuracy,
            "sample_sd_accuracy": _sample_sd(accuracies),
            "mean_macro_f1": mean_macro_f1,
            "sample_sd_macro_f1": _sample_sd(macro_f1s),
            "delta_vs_historical_accuracy": mean_accuracy - baseline["accuracy"],
            "delta_vs_historical_macro_f1": mean_macro_f1 - baseline["macro_f1"],
        }
    if len(provenance_levels) > 1:
        raise ValueError("Cannot combine legacy and strictly signed checkpoints in one comparison")
    report["protocol"]["checkpoint_provenance"] = sorted(provenance_levels)
    if maintained_common_signature is not None:
        report["protocol"]["maintained_common_run_signature"] = maintained_common_signature
    save_json(output_path, report)
    return output_path


def audit_inputs(
    train_manifest: Path,
    real_cache_root: Path,
    generated_cache_root: Path,
    config: SpectrogramConfig,
    ratios: Iterable[int],
    generators: Iterable[str],
) -> dict[str, Any]:
    ratios = tuple(dict.fromkeys(int(value) for value in ratios))
    generators = tuple(dict.fromkeys(str(value) for value in generators))
    _, real_cache_audit = audit_spectrogram_cache(real_cache_root, config)
    real = build_real_dataset(train_manifest, real_cache_root, config, training=True)
    result: dict[str, Any] = {
        "real_rows": len(real),
        "real_cache": str(real_cache_root.resolve()),
        "real_cache_audit": real_cache_audit,
        "pools": {},
    }
    for generator_name in generators:
        manifest = generated_cache_root / generator_name / "manifest.csv"
        pool_audit = audit_generated_pool(manifest, generated_cache_root, DEFAULT_CLASSES, config)
        result["pools"][generator_name] = {"audit": pool_audit, "ratios": {}}
        for ratio in ratios:
            dataset = GeneratedSpectrogramDataset(
                manifest,
                generated_cache_root,
                DEFAULT_CLASSES,
                config,
                int(ratio),
                training=True,
            )
            spec, _, path = dataset[0]
            result["pools"][generator_name]["ratios"][str(ratio)] = {
                "rows": len(dataset),
                "first_shape": list(spec.shape),
                "first_path": path,
            }
    return result
