from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from bird_song.augmentation.data import GeneratedSpectrogramDataset, audit_generated_pool
from bird_song.augmentation.experiment import evaluate_model, select_ratio
from bird_song.classifier.model import build_classifier, count_trainable_parameters
from bird_song.config import DEFAULT_CLASSES, SpectrogramConfig
from bird_song.data import ManifestDataset
from bird_song.generation.checkpoint_evaluation import audit_pools
from bird_song.runtime import atomic_torch_save, load_checkpoint, save_json, seed_everything
from bird_song.spectrogram_cache import audit_spectrogram_cache, sha256_file


GENERATORS = ("vae_v3", "diffusion")
DEFAULT_REAL_PER_SPECIES = 50
DEFAULT_RATIOS = (50, 100, 200)
DEFAULT_REAL_SUBSET_SEEDS = (101, 202, 303)
DEFAULT_TRAIN_SEEDS = (42, 123, 777)
DEFAULT_POOL_SEEDS = (42, 123, 777)
DEFAULT_STEPS = 1_440
DEFAULT_VALIDATE_EVERY = 36
DEFAULT_BATCH_SIZE = 64
PROTOCOL_VERSION = 1
MODEL_WIDTH = 32
MODEL_DROPOUT = 0.30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 5.0
RATIO_TIE_TOLERANCE = 0.002
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class ReplicateBlock:
    real_subset_seed: int
    train_seed: int
    pool_seed: int


@dataclass(frozen=True)
class ExperimentCondition:
    condition: str
    generator: str | None
    ratio_per_species: int


@dataclass(frozen=True)
class PoolReference:
    model: str
    seed: int
    manifest: Path
    cache_root: Path
    generation_metadata: Path


def _portable(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sample_sd(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int = 20260810,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("Bootstrap values must be a non-empty one-dimensional sequence")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def experiment_conditions(ratios: Iterable[int] = DEFAULT_RATIOS) -> tuple[ExperimentCondition, ...]:
    requested = tuple(dict.fromkeys(int(value) for value in ratios))
    if not requested or any(value < 1 for value in requested):
        raise ValueError("At least one positive generated ratio is required")
    return (
        ExperimentCondition("real_only", None, 0),
        *(
            ExperimentCondition(f"{generator}_plus_{ratio}", generator, ratio)
            for generator in GENERATORS
            for ratio in requested
        ),
    )


def replicate_blocks(
    real_subset_seeds: Sequence[int] = DEFAULT_REAL_SUBSET_SEEDS,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    pool_seeds: Sequence[int] = DEFAULT_POOL_SEEDS,
) -> tuple[ReplicateBlock, ...]:
    subset_values = tuple(dict.fromkeys(int(value) for value in real_subset_seeds))
    train_values = tuple(dict.fromkeys(int(value) for value in train_seeds))
    pool_values = tuple(dict.fromkeys(int(value) for value in pool_seeds))
    if not subset_values or not train_values or not pool_values:
        raise ValueError("Real-subset, training, and pool seed lists must be non-empty")
    return tuple(
        ReplicateBlock(
            real_subset_seed=subset_seed,
            train_seed=train_seed,
            pool_seed=pool_values[(subset_index + train_index) % len(pool_values)],
        )
        for subset_index, subset_seed in enumerate(subset_values)
        for train_index, train_seed in enumerate(train_values)
    )


def select_real_subset_rows(
    rows: pd.DataFrame,
    *,
    real_per_species: int = DEFAULT_REAL_PER_SPECIES,
    seed: int,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> pd.DataFrame:
    """Choose one clip from each of N distinct recording IDs per species."""
    if real_per_species < 1:
        raise ValueError("real_per_species must be positive")
    required = {"split", "name", "id", "relative_wav_path", "audio_sha256"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Training manifest is missing low-resource columns: {sorted(missing)}")
    if rows.empty or rows[list(required)].isna().any().any():
        raise ValueError("Training manifest is empty or has missing low-resource values")
    unknown = sorted(set(rows["name"].astype(str)) - set(classes))
    if unknown:
        raise ValueError(f"Training manifest has unknown classes: {unknown}")

    selected_frames: list[pd.DataFrame] = []
    for class_index, species in enumerate(classes):
        group = rows[rows["name"] == species].copy()
        group["id"] = group["id"].astype(str)
        recording_ids = np.asarray(sorted(group["id"].unique()), dtype=object)
        if len(recording_ids) < real_per_species:
            raise ValueError(
                f"{species} has only {len(recording_ids)} distinct recording IDs; "
                f"cannot select {real_per_species}"
            )
        rng = np.random.default_rng(int(seed) + 10_007 * (class_index + 1))
        chosen_ids = rng.permutation(recording_ids)[:real_per_species]
        species_rows: list[pd.Series] = []
        for rank, recording_id in enumerate(chosen_ids):
            candidates = group[group["id"] == str(recording_id)].sort_values(
                ["relative_wav_path", "audio_sha256"], kind="stable"
            )
            chosen_index = int(rng.integers(0, len(candidates)))
            chosen = candidates.iloc[chosen_index].copy()
            chosen["low_resource_recording_rank"] = rank
            species_rows.append(chosen)
        selected_frames.append(pd.DataFrame(species_rows))

    selected = pd.concat(selected_frames, ignore_index=True)
    selected["low_resource_subset_seed"] = int(seed)
    selected["low_resource_real_per_species"] = int(real_per_species)
    counts = selected["name"].value_counts().reindex(classes, fill_value=0)
    unique_ids = selected.groupby("name")["id"].nunique().reindex(classes, fill_value=0)
    if not (counts == real_per_species).all() or not (unique_ids == real_per_species).all():
        raise AssertionError("Low-resource subset is not balanced across distinct recordings")
    if selected["audio_sha256"].astype(str).str.upper().duplicated().any():
        raise ValueError("Low-resource subset unexpectedly contains duplicate audio content")
    return selected.sort_values(["name", "low_resource_recording_rank"], kind="stable").reset_index(drop=True)


def prepare_real_subsets(
    source_manifest: Path,
    output_dir: Path,
    *,
    real_per_species: int = DEFAULT_REAL_PER_SPECIES,
    seeds: Sequence[int] = DEFAULT_REAL_SUBSET_SEEDS,
) -> dict[int, Path]:
    source_rows = pd.read_csv(source_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    for seed in tuple(dict.fromkeys(int(value) for value in seeds)):
        selected = select_real_subset_rows(
            source_rows,
            real_per_species=real_per_species,
            seed=seed,
        )
        path = output_dir / f"real_{real_per_species}_subset_{seed}.csv"
        expected = selected.to_csv(index=False, lineterminator="\n")
        if path.is_file() and path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Existing low-resource subset differs from deterministic selection: {path}")
        if not path.is_file():
            path.write_text(expected, encoding="utf-8")
        result[seed] = path
    return result


def pool_reference(project_root: Path, pool_root: Path, model: str, seed: int) -> PoolReference:
    if model not in GENERATORS:
        raise ValueError(f"Unknown generator {model!r}; choose from {GENERATORS}")
    if int(seed) == 42:
        cache_root = project_root / "artifacts/generated_spectrograms"
        model_root = cache_root / model
    else:
        model_root = pool_root / model / f"seed_{int(seed)}"
        cache_root = model_root
    return PoolReference(
        model=model,
        seed=int(seed),
        manifest=model_root / "manifest.csv",
        cache_root=cache_root,
        generation_metadata=model_root / "generation.json",
    )


def condition_run_dir(run_root: Path, block: ReplicateBlock, condition: ExperimentCondition) -> Path:
    base = run_root / f"subset_{block.real_subset_seed}" / f"train_{block.train_seed}"
    if condition.generator is None:
        return base / "real_only"
    return (
        base
        / condition.generator
        / f"pool_{block.pool_seed}"
        / f"ratio_{condition.ratio_per_species}"
    )


class SharedSpectrogramMask(Dataset[tuple[torch.Tensor, int, str]]):
    """Apply the same post-cache masking policy to real and generated rows."""

    def __init__(self, dataset: Dataset[tuple[torch.Tensor, int, str]], *, training: bool) -> None:
        self.dataset = dataset
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def mask(spec: torch.Tensor) -> torch.Tensor:
        result = spec.clone()
        if torch.rand(()) < 0.5:
            height = int(result.shape[-2])
            width = int(torch.randint(0, min(12, height) + 1, ()).item())
            if width:
                start = int(torch.randint(0, height - width + 1, ()).item())
                result[:, start : start + width, :] = -1.0
        if torch.rand(()) < 0.5:
            length = int(result.shape[-1])
            width = int(torch.randint(0, min(16, length) + 1, ()).item())
            if width:
                start = int(torch.randint(0, length - width + 1, ()).item())
                result[:, :, start : start + width] = -1.0
        return result

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        spec, label, path = self.dataset[index]
        return (self.mask(spec) if self.training else spec), label, path


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    training: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
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


def _real_dataset(
    manifest: Path,
    cache_root: Path,
    config: SpectrogramConfig,
    *,
    training: bool,
) -> SharedSpectrogramMask:
    base = ManifestDataset(
        manifest,
        None,
        DEFAULT_CLASSES,
        config,
        training=False,
        spectrogram_cache_root=cache_root,
    )
    return SharedSpectrogramMask(base, training=training)


def _generated_dataset(
    reference: PoolReference,
    config: SpectrogramConfig,
    ratio_per_species: int,
    *,
    training: bool,
) -> SharedSpectrogramMask:
    base = GeneratedSpectrogramDataset(
        reference.manifest,
        reference.cache_root,
        DEFAULT_CLASSES,
        config,
        ratio_per_species,
        training=False,
    )
    return SharedSpectrogramMask(base, training=training)


def _training_protocol() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": "crnn",
        "model_width": MODEL_WIDTH,
        "model_dropout": MODEL_DROPOUT,
        "initialization": "from scratch for every replicate block and condition",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR over fixed optimizer steps",
        "label_smoothing": LABEL_SMOOTHING,
        "gradient_clip_max_norm": GRAD_CLIP_NORM,
        "input_transform": (
            "identical for real and generated training rows: independent p=0.5 frequency mask "
            "up to 12 bins and p=0.5 time mask up to 16 frames"
        ),
        "validation_transform": "none",
        "selection_metric": "best real-validation macro_f1 within each run",
    }


def _run_signature(
    *,
    project_root: Path,
    block: ReplicateBlock,
    condition: ExperimentCondition,
    source_train_manifest: Path,
    subset_manifest: Path,
    validation_manifest: Path,
    cache_root: Path,
    spectrogram_config_path: Path,
    config: SpectrogramConfig,
    pool: PoolReference | None,
    steps: int,
    validate_every: int,
    batch_size: int,
    workers: int,
    real_per_species: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "condition": condition.condition,
        "generator": condition.generator,
        "ratio_per_species": condition.ratio_per_species,
        "real_per_species": int(real_per_species),
        "real_subset_seed": block.real_subset_seed,
        "train_seed": block.train_seed,
        "pool_seed": block.pool_seed if condition.generator else None,
        "steps": int(steps),
        "validate_every": int(validate_every),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "device_type": device.type,
        "classes": list(DEFAULT_CLASSES),
        "source_train_manifest": _portable(source_train_manifest, project_root),
        "source_train_manifest_sha256": sha256_file(source_train_manifest),
        "subset_manifest": _portable(subset_manifest, project_root),
        "subset_manifest_sha256": sha256_file(subset_manifest),
        "validation_manifest": _portable(validation_manifest, project_root),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "real_cache_manifest_sha256": sha256_file(cache_root / "spectrogram_manifest.csv"),
        "pool_manifest": _portable(pool.manifest, project_root) if pool else None,
        "pool_manifest_sha256": sha256_file(pool.manifest) if pool else None,
        "pool_generation_sha256": (
            sha256_file(pool.generation_metadata) if pool and pool.generation_metadata.is_file() else None
        ),
        "spectrogram_config_sha256": sha256_file(spectrogram_config_path),
        "spectrogram_config": config.to_dict(),
        "training_protocol": _training_protocol(),
    }


def _validate_existing_run(output_dir: Path, expected_signature: dict[str, Any]) -> Path:
    checkpoint_path = output_dir / "best.pt"
    config_path = output_dir / "config.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"Existing low-resource run is incomplete: {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if checkpoint.get("run_signature") != expected_signature or run_config.get("run_signature") != expected_signature:
        raise RuntimeError(
            f"Existing low-resource run is incompatible with the requested protocol: {output_dir}. "
            "Use a fresh run root or pass --overwrite intentionally."
        )
    return checkpoint_path


def train_condition(
    *,
    project_root: Path,
    block: ReplicateBlock,
    condition: ExperimentCondition,
    source_train_manifest: Path,
    subset_manifest: Path,
    validation_manifest: Path,
    cache_root: Path,
    pool_root: Path,
    spectrogram_config_path: Path,
    output_dir: Path,
    config: SpectrogramConfig,
    device: torch.device,
    real_per_species: int = DEFAULT_REAL_PER_SPECIES,
    steps: int = DEFAULT_STEPS,
    validate_every: int = DEFAULT_VALIDATE_EVERY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 0,
    overwrite: bool = False,
) -> Path:
    if steps < 1 or validate_every < 1 or steps % validate_every:
        raise ValueError("steps and validate_every must be positive, and steps must be divisible")
    pool = (
        pool_reference(project_root, pool_root, condition.generator, block.pool_seed)
        if condition.generator
        else None
    )
    signature = _run_signature(
        project_root=project_root,
        block=block,
        condition=condition,
        source_train_manifest=source_train_manifest,
        subset_manifest=subset_manifest,
        validation_manifest=validation_manifest,
        cache_root=cache_root,
        spectrogram_config_path=spectrogram_config_path,
        config=config,
        pool=pool,
        steps=steps,
        validate_every=validate_every,
        batch_size=batch_size,
        workers=workers,
        real_per_species=real_per_species,
        device=device,
    )
    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.exists() and not overwrite:
        return _validate_existing_run(output_dir, signature)

    seed_everything(block.train_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_data = _real_dataset(subset_manifest, cache_root, config, training=True)
    datasets: list[Dataset] = [real_data]
    generated_count = 0
    if pool is not None:
        generated_data = _generated_dataset(
            pool,
            config,
            condition.ratio_per_species,
            training=True,
        )
        datasets.append(generated_data)
        generated_count = len(generated_data)
    train_data: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    validation_data = _real_dataset(validation_manifest, cache_root, config, training=False)
    train_loader = _loader(
        train_data,
        batch_size=batch_size,
        workers=workers,
        training=True,
        seed=block.train_seed,
    )
    validation_loader = _loader(
        validation_data,
        batch_size=batch_size,
        workers=workers,
        training=False,
        seed=0,
    )

    model = build_classifier(
        "crnn",
        num_classes=len(DEFAULT_CLASSES),
        dropout=MODEL_DROPOUT,
        width=MODEL_WIDTH,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    history: list[dict[str, Any]] = []
    best_f1 = -math.inf
    best_step = 0
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
                    "format_version": 4,
                    "model_state": model.state_dict(),
                    "model_config": model.metadata(),
                    "architecture": "crnn",
                    "trainable_parameters": count_trainable_parameters(model),
                    "classes": list(DEFAULT_CLASSES),
                    "spectrogram_config": config.to_dict(),
                    "condition": condition.condition,
                    "generator": condition.generator,
                    "ratio": condition.ratio_per_species,
                    "real_per_species": real_per_species,
                    "real_subset_seed": block.real_subset_seed,
                    "train_seed": block.train_seed,
                    "pool_seed": block.pool_seed if condition.generator else None,
                    "steps": steps,
                    "validation_every": validate_every,
                    "best_step": best_step,
                    "best_validation_macro_f1": best_f1,
                    "validation": validation,
                    "run_signature": signature,
                },
                checkpoint_path,
            )
        if step % (validate_every * 4) == 0:
            print(
                f"condition={condition.condition} subset={block.real_subset_seed} "
                f"train={block.train_seed} pool={block.pool_seed if condition.generator else '-'} "
                f"step={step} val_macro_f1={current_f1:.4f}",
                flush=True,
            )

    real_count = len(real_data)
    save_json(
        output_dir / "config.json",
        {
            "condition": condition.condition,
            "generator": condition.generator,
            "ratio_per_species": condition.ratio_per_species,
            "real_per_species": real_per_species,
            "real_subset_seed": block.real_subset_seed,
            "train_seed": block.train_seed,
            "pool_seed": block.pool_seed if condition.generator else None,
            "device": str(device),
            "steps": steps,
            "validation_every": validate_every,
            "batch_size": batch_size,
            "workers": workers,
            "real_rows": real_count,
            "generated_rows": generated_count,
            "training_rows": real_count + generated_count,
            "synthetic_share": generated_count / (real_count + generated_count),
            "elapsed_seconds": time.perf_counter() - started,
            "best_step": best_step,
            "best_validation_macro_f1": best_f1,
            "run_signature": signature,
        },
    )
    save_json(output_dir / "history.json", history)
    return checkpoint_path


def _content_safe_audit(
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
) -> dict[str, Any]:
    frames = {
        "train": pd.read_csv(train_manifest),
        "validation": pd.read_csv(validation_manifest),
        "test": pd.read_csv(test_manifest),
    }
    required = {"split", "name", "id", "relative_wav_path", "audio_sha256"}
    for split, frame in frames.items():
        missing = required - set(frame.columns)
        if missing or frame.empty or frame[list(required)].isna().any().any():
            raise ValueError(f"{split} manifest is incomplete: missing={sorted(missing)}")
        if set(frame["name"]) != set(DEFAULT_CLASSES):
            raise ValueError(f"{split} manifest class set does not match the classifier")
    split_hashes = {
        split: set(frame["audio_sha256"].astype(str).str.upper()) for split, frame in frames.items()
    }
    overlaps = {
        f"{left}_vs_{right}": len(split_hashes[left] & split_hashes[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"Content-safe manifests contain exact-content overlap: {overlaps}")
    return {
        "rows": {split: int(len(frame)) for split, frame in frames.items()},
        "recording_ids_per_species": {
            split: {
                species: int(count)
                for species, count in frame.groupby("name")["id"].nunique().items()
            }
            for split, frame in frames.items()
        },
        "exact_content_overlap": overlaps,
    }


def audit_inputs(
    *,
    project_root: Path,
    run_root: Path,
    source_train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    cache_root: Path,
    pool_root: Path,
    spectrogram_config_path: Path,
    real_per_species: int = DEFAULT_REAL_PER_SPECIES,
    ratios: Sequence[int] = DEFAULT_RATIOS,
    real_subset_seeds: Sequence[int] = DEFAULT_REAL_SUBSET_SEEDS,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    pool_seeds: Sequence[int] = DEFAULT_POOL_SEEDS,
) -> dict[str, Any]:
    requested_ratios = tuple(dict.fromkeys(int(value) for value in ratios))
    if max(requested_ratios) > 200:
        raise ValueError("Existing generated pools support at most 200 rows per species and pool seed")
    config = SpectrogramConfig.from_json(spectrogram_config_path)
    content_safe = _content_safe_audit(source_train_manifest, validation_manifest, test_manifest)
    _, cache_audit = audit_spectrogram_cache(cache_root, config)
    for manifest in (source_train_manifest, validation_manifest, test_manifest):
        ManifestDataset(
            manifest,
            None,
            DEFAULT_CLASSES,
            config,
            training=False,
            spectrogram_cache_root=cache_root,
        )
    subsets = prepare_real_subsets(
        source_train_manifest,
        run_root / "manifests",
        real_per_species=real_per_species,
        seeds=real_subset_seeds,
    )
    subset_summary = []
    for seed, path in subsets.items():
        frame = pd.read_csv(path)
        subset_summary.append(
            {
                "seed": seed,
                "path": _portable(path, project_root),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                "rows_per_species": {
                    species: int(count) for species, count in frame["name"].value_counts().items()
                },
                "unique_recordings_per_species": {
                    species: int(count) for species, count in frame.groupby("name")["id"].nunique().items()
                },
            }
        )

    pool_audit = audit_pools(project_root, pool_root, expected_samples=200, seeds=pool_seeds)
    all_hashes: list[str] = []
    pool_details: list[dict[str, Any]] = []
    for model in GENERATORS:
        for seed in pool_seeds:
            reference = pool_reference(project_root, pool_root, model, int(seed))
            details = audit_generated_pool(
                reference.manifest,
                reference.cache_root,
                DEFAULT_CLASSES,
                config,
            )
            frame = pd.read_csv(reference.manifest)
            all_hashes.extend(frame["array_sha256"].astype(str).str.upper().tolist())
            pool_details.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "manifest": _portable(reference.manifest, project_root),
                    "manifest_sha256": sha256_file(reference.manifest),
                    "rows": details["rows"],
                    "rows_per_species": details["rows_per_species"],
                }
            )
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError("Generated pools contain duplicate arrays across model/seed boundaries")

    blocks = replicate_blocks(real_subset_seeds, train_seeds, pool_seeds)
    conditions = experiment_conditions(requested_ratios)
    return {
        "schema_version": 1,
        "content_safe": content_safe,
        "cache": {
            "logical_rows": int(cache_audit["row_count"]),
            "physical_arrays": int(cache_audit["unique_object_count"]),
        },
        "real_per_species": int(real_per_species),
        "ratios_per_species": list(requested_ratios),
        "synthetic_shares": {
            str(ratio): (ratio * len(DEFAULT_CLASSES))
            / (real_per_species * len(DEFAULT_CLASSES) + ratio * len(DEFAULT_CLASSES))
            for ratio in requested_ratios
        },
        "subsets": subset_summary,
        "pool_audit": pool_audit,
        "pool_details": pool_details,
        "unique_generated_arrays": len(set(all_hashes)),
        "replicate_blocks": [block.__dict__ for block in blocks],
        "conditions": [condition.__dict__ for condition in conditions],
        "expected_training_runs": len(blocks) * len(conditions),
    }


def run_sweep(
    *,
    project_root: Path,
    run_root: Path,
    source_train_manifest: Path,
    validation_manifest: Path,
    cache_root: Path,
    pool_root: Path,
    spectrogram_config_path: Path,
    config: SpectrogramConfig,
    device: torch.device,
    real_per_species: int = DEFAULT_REAL_PER_SPECIES,
    ratios: Sequence[int] = DEFAULT_RATIOS,
    real_subset_seeds: Sequence[int] = DEFAULT_REAL_SUBSET_SEEDS,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    pool_seeds: Sequence[int] = DEFAULT_POOL_SEEDS,
    steps: int = DEFAULT_STEPS,
    validate_every: int = DEFAULT_VALIDATE_EVERY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 0,
    overwrite: bool = False,
) -> list[Path]:
    subsets = prepare_real_subsets(
        source_train_manifest,
        run_root / "manifests",
        real_per_species=real_per_species,
        seeds=real_subset_seeds,
    )
    blocks = replicate_blocks(real_subset_seeds, train_seeds, pool_seeds)
    conditions = experiment_conditions(ratios)
    checkpoints: list[Path] = []
    total = len(blocks) * len(conditions)
    run_index = 0
    for block in blocks:
        subset_manifest = subsets[block.real_subset_seed]
        for condition in conditions:
            run_index += 1
            output_dir = condition_run_dir(run_root, block, condition)
            print(f"start [{run_index}/{total}] {output_dir.relative_to(run_root)} device={device}", flush=True)
            checkpoint = train_condition(
                project_root=project_root,
                block=block,
                condition=condition,
                source_train_manifest=source_train_manifest,
                subset_manifest=subset_manifest,
                validation_manifest=validation_manifest,
                cache_root=cache_root,
                pool_root=pool_root,
                spectrogram_config_path=spectrogram_config_path,
                output_dir=output_dir,
                config=config,
                device=device,
                real_per_species=real_per_species,
                steps=steps,
                validate_every=validate_every,
                batch_size=batch_size,
                workers=workers,
                overwrite=overwrite,
            )
            checkpoints.append(checkpoint)
            print(f"complete checkpoint={checkpoint}", flush=True)
    return checkpoints


def _checkpoint_payload(path: Path, block: ReplicateBlock, condition: ExperimentCondition) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Low-resource checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "condition": condition.condition,
        "generator": condition.generator,
        "ratio": condition.ratio_per_species,
        "real_subset_seed": block.real_subset_seed,
        "train_seed": block.train_seed,
        "pool_seed": block.pool_seed if condition.generator else None,
    }
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatched or payload.get("run_signature") is None:
        raise ValueError(f"Low-resource checkpoint metadata mismatch ({mismatched}): {path}")
    return payload


def collect_validation_summary(
    *,
    project_root: Path,
    run_root: Path,
    ratios: Sequence[int] = DEFAULT_RATIOS,
    real_subset_seeds: Sequence[int] = DEFAULT_REAL_SUBSET_SEEDS,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    pool_seeds: Sequence[int] = DEFAULT_POOL_SEEDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block in replicate_blocks(real_subset_seeds, train_seeds, pool_seeds):
        for condition in experiment_conditions(ratios):
            path = condition_run_dir(run_root, block, condition) / "best.pt"
            payload = _checkpoint_payload(path, block, condition)
            validation = payload["validation"]
            rows.append(
                {
                    "condition": condition.condition,
                    "generator": condition.generator or "none",
                    "ratio_per_species": condition.ratio_per_species,
                    "real_subset_seed": block.real_subset_seed,
                    "train_seed": block.train_seed,
                    "pool_seed": block.pool_seed if condition.generator else np.nan,
                    "best_step": int(payload["best_step"]),
                    "validation_accuracy": float(validation["accuracy"]),
                    "validation_macro_f1": float(validation["macro_f1"]),
                    **{
                        f"recall_{species.lower().replace(' ', '_')}": float(value)
                        for species, value in validation["per_species_recall"].items()
                    },
                    "checkpoint": _portable(path, project_root),
                    "checkpoint_sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "macro_f1",
        "recall_american_robin",
        "recall_northern_cardinal",
        "recall_song_sparrow",
        "minimum_species_recall",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values, strict=True))
        row["runs"] = int(len(group))
        for metric in metric_columns:
            values = group[metric].astype(float).tolist()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sample_sd"] = _sample_sd(values)
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def select_and_evaluate(
    *,
    project_root: Path,
    run_root: Path,
    test_manifest: Path,
    cache_root: Path,
    config: SpectrogramConfig,
    device: torch.device,
    output_path: Path,
    real_per_species: int = DEFAULT_REAL_PER_SPECIES,
    ratios: Sequence[int] = DEFAULT_RATIOS,
    real_subset_seeds: Sequence[int] = DEFAULT_REAL_SUBSET_SEEDS,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    pool_seeds: Sequence[int] = DEFAULT_POOL_SEEDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 0,
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Evaluation already exists: {output_path}; pass --overwrite-report intentionally")
    validation = collect_validation_summary(
        project_root=project_root,
        run_root=run_root,
        ratios=ratios,
        real_subset_seeds=real_subset_seeds,
        train_seeds=train_seeds,
        pool_seeds=pool_seeds,
    )
    selections: dict[str, int] = {}
    selection_rows: list[dict[str, Any]] = []
    for generator in GENERATORS:
        generator_rows = validation[validation["generator"] == generator]
        means = {
            int(ratio): float(group["validation_macro_f1"].mean())
            for ratio, group in generator_rows.groupby("ratio_per_species")
        }
        selected = select_ratio(means, tie_tolerance=RATIO_TIE_TOLERANCE)
        selections[generator] = selected
        for ratio, group in generator_rows.groupby("ratio_per_species"):
            values = group["validation_macro_f1"].astype(float).tolist()
            selection_rows.append(
                {
                    "generator": generator,
                    "ratio_per_species": int(ratio),
                    "runs": len(values),
                    "validation_macro_f1_mean": float(np.mean(values)),
                    "validation_macro_f1_sample_sd": _sample_sd(values),
                    "validation_macro_f1_min": float(np.min(values)),
                    "validation_macro_f1_max": float(np.max(values)),
                    "selected_for_test": int(ratio) == selected,
                }
            )

    test_data = _real_dataset(test_manifest, cache_root, config, training=False)
    test_loader = _loader(test_data, batch_size=batch_size, workers=workers, training=False, seed=0)
    test_rows: list[dict[str, Any]] = []
    confusion_records: list[dict[str, Any]] = []
    selected_conditions = (
        ExperimentCondition("real_only", None, 0),
        *(
            ExperimentCondition(f"{generator}_plus_{selections[generator]}", generator, selections[generator])
            for generator in GENERATORS
        ),
    )
    for block in replicate_blocks(real_subset_seeds, train_seeds, pool_seeds):
        for condition in selected_conditions:
            path = condition_run_dir(run_root, block, condition) / "best.pt"
            _checkpoint_payload(path, block, condition)
            model, classes, loaded_config, payload = load_checkpoint(path, device)
            if classes != DEFAULT_CLASSES or loaded_config != config:
                raise ValueError(f"Classifier contract mismatch: {path}")
            result = evaluate_model(model, test_loader, device)
            recalls = result["per_species_recall"]
            row = {
                "condition": condition.condition,
                "generator": condition.generator or "none",
                "ratio_per_species": condition.ratio_per_species,
                "real_per_species": real_per_species,
                "real_subset_seed": block.real_subset_seed,
                "train_seed": block.train_seed,
                "pool_seed": block.pool_seed if condition.generator else np.nan,
                "accuracy": float(result["accuracy"]),
                "macro_f1": float(result["macro_f1"]),
                "recall_american_robin": float(recalls["American Robin"]),
                "recall_northern_cardinal": float(recalls["Northern Cardinal"]),
                "recall_song_sparrow": float(recalls["Song Sparrow"]),
                "minimum_species_recall": float(min(recalls.values())),
                "sample_count": int(result["sample_count"]),
                "checkpoint": _portable(path, project_root),
                "checkpoint_sha256": sha256_file(path),
                "best_validation_macro_f1": float(payload["best_validation_macro_f1"]),
            }
            test_rows.append(row)
            confusion_records.append(
                {
                    "condition": condition.condition,
                    "real_subset_seed": block.real_subset_seed,
                    "train_seed": block.train_seed,
                    "pool_seed": block.pool_seed if condition.generator else None,
                    "classes": list(DEFAULT_CLASSES),
                    "confusion": result["confusion"],
                }
            )

    test_per_run = pd.DataFrame(test_rows)
    test_aggregate = _aggregate_metrics(
        test_per_run,
        ["condition", "generator", "ratio_per_species", "real_per_species"],
    )
    baseline = test_per_run[test_per_run["condition"] == "real_only"].copy()
    paired_frames: list[pd.DataFrame] = []
    for generator in GENERATORS:
        selected_name = f"{generator}_plus_{selections[generator]}"
        treatment = test_per_run[test_per_run["condition"] == selected_name].copy()
        merged = treatment.merge(
            baseline,
            on=["real_subset_seed", "train_seed"],
            suffixes=("_synthetic", "_baseline"),
            validate="one_to_one",
        )
        paired = pd.DataFrame(
            {
                "generator": generator,
                "selected_ratio_per_species": selections[generator],
                "real_subset_seed": merged["real_subset_seed"],
                "train_seed": merged["train_seed"],
                "pool_seed": merged["pool_seed_synthetic"],
            }
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "recall_american_robin",
            "recall_northern_cardinal",
            "recall_song_sparrow",
            "minimum_species_recall",
        ):
            paired[f"{metric}_baseline"] = merged[f"{metric}_baseline"]
            paired[f"{metric}_synthetic"] = merged[f"{metric}_synthetic"]
            paired[f"delta_{metric}"] = (
                merged[f"{metric}_synthetic"] - merged[f"{metric}_baseline"]
            )
        paired_frames.append(paired)
    paired_deltas = pd.concat(paired_frames, ignore_index=True)

    paired_summary: list[dict[str, Any]] = []
    for generator, group in paired_deltas.groupby("generator"):
        row: dict[str, Any] = {
            "generator": generator,
            "selected_ratio_per_species": int(group["selected_ratio_per_species"].iloc[0]),
            "paired_blocks": int(len(group)),
        }
        for metric in ("accuracy", "macro_f1", "minimum_species_recall"):
            values = group[f"delta_{metric}"].astype(float).tolist()
            low, high = _bootstrap_mean_interval(values)
            row[f"delta_{metric}_mean"] = float(np.mean(values))
            row[f"delta_{metric}_sample_sd"] = _sample_sd(values)
            row[f"delta_{metric}_min"] = float(np.min(values))
            row[f"delta_{metric}_max"] = float(np.max(values))
            row[f"delta_{metric}_bootstrap_95_low"] = low
            row[f"delta_{metric}_bootstrap_95_high"] = high
            row[f"delta_{metric}_positive_blocks"] = int(np.sum(np.asarray(values) > 0))
        paired_summary.append(row)

    evaluation_dir = output_path.parent / "evaluation_tables"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(evaluation_dir / "validation_per_run.csv", index=False, lineterminator="\n")
    pd.DataFrame(selection_rows).to_csv(
        evaluation_dir / "validation_selection.csv", index=False, lineterminator="\n"
    )
    test_per_run.to_csv(evaluation_dir / "test_per_run.csv", index=False, lineterminator="\n")
    test_aggregate.to_csv(evaluation_dir / "test_aggregate.csv", index=False, lineterminator="\n")
    paired_deltas.to_csv(evaluation_dir / "paired_deltas.csv", index=False, lineterminator="\n")

    blocks = replicate_blocks(real_subset_seeds, train_seeds, pool_seeds)
    block_count = len(blocks)
    payload = {
        "schema_version": 1,
        "title": "Low-resource CRNN synthetic-augmentation evaluation",
        "protocol": {
            "question": (
                "When CRNN training is restricted to 50 labeled real spectrograms per species, "
                "does adding pretrained generated spectrograms improve held-out real classification?"
            ),
            "claim_scope": "simulated classifier-label scarcity with access to pretrained generators",
            "real_per_species": real_per_species,
            "ratios_per_species": list(dict.fromkeys(int(value) for value in ratios)),
            "real_subset_seeds": list(dict.fromkeys(int(value) for value in real_subset_seeds)),
            "train_seeds": list(dict.fromkeys(int(value) for value in train_seeds)),
            "pool_seeds": list(dict.fromkeys(int(value) for value in pool_seeds)),
            "replicate_blocks": [block.__dict__ for block in blocks],
            "selection_metric": (
                f"mean best real-validation macro_f1 across {block_count} matched blocks; "
                "choose the smallest "
                f"ratio within {RATIO_TIE_TOLERANCE:.3f} of the best"
            ),
            "test_policy": "evaluate real-only and one validation-selected ratio per generator",
            "paired_interval": (
                f"{BOOTSTRAP_RESAMPLES} deterministic bootstrap resamples over "
                f"{block_count} matched blocks"
            ),
            "training_protocol": _training_protocol(),
        },
        "selected_ratios": selections,
        "validation_selection": selection_rows,
        "test_per_run": test_rows,
        "test_aggregate": test_aggregate.to_dict(orient="records"),
        "paired_deltas": paired_deltas.to_dict(orient="records"),
        "paired_summary": paired_summary,
        "confusion_matrices": confusion_records,
        "caveats": [
            "The three evaluated birds are a simulated low-resource setting, not genuinely rare species.",
            "The classifiers receive only 50 labeled real rows per species, but the pretrained generators saw more source data.",
            "The current 489-clip project test set has prior evaluation history and is not a newly acquired external holdout.",
            (
                "Real-subset seeds are repeated subsamples of the same source split and necessarily "
                "overlap, so the matched blocks are not independent external replications."
            ),
            (
                f"{block_count} paired blocks support a descriptive stability analysis but do not "
                "separate every interaction among subset, training, and pool seeds."
            ),
            (
                "The selected 200-per-species ratio is the largest amount available in the existing "
                "pools, so it is the best tested ratio rather than an estimated optimum."
            ),
        ],
    }
    save_json(output_path, payload)
    return output_path
