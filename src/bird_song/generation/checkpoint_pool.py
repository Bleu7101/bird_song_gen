"""Resumable, deterministic classifier-input pools for frozen generator checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .checkpoint_models import (
    DIFFUSION_CLAMP,
    DIFFUSION_DDIM_ETA,
    DIFFUSION_DDIM_STEPS,
    DIFFUSION_GUIDANCE,
    DIFFUSION_PARAMETER_COUNT,
    DIFFUSION_TIMESTEPS,
    GENERATOR_CLASSES,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    VAE_PARAMETER_COUNT,
    VAE_TEMPERATURE,
    ConditionalVAE,
    build_diffusion_schedule,
    checkpoint_parameter_count,
    classifier_scale_from_standardized,
    ddim_sample,
    load_diffusion_model,
    load_vae_model,
)


GENERATOR_LABEL_TO_ID = {name: index for index, name in enumerate(GENERATOR_CLASSES)}


def species_slug(species: str) -> str:
    return species.lower().replace(" ", "_").replace("'", "")


def deterministic_sample_seed(seed: int, label_id: int, sample_index: int) -> int:
    """Derive a stable per-sample stream without relying on loop/chunk order."""
    return int(seed) * 1_000_003 + int(label_id) * 100_003 + int(sample_index) + 17


def verify_checkpoint(path: Path, model: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if model not in {"vae_v3", "diffusion"}:
        raise ValueError(f"unknown generator model: {model}")


def _atomic_dataframe_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _valid_classifier_array(path: Path) -> np.ndarray | None:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if array.shape != (1, 128, 128) or array.dtype.kind not in "fc" or not np.isfinite(array).all():
        return None
    if float(array.min()) < -1.00001 or float(array.max()) > 1.00001:
        return None
    return array.astype(np.float32, copy=False)


def _load_posterior_bank(path: Path, device: torch.device) -> dict[int, dict[str, Any]]:
    bank_package = torch.load(path, map_location="cpu", weights_only=True)
    label_to_id = dict(bank_package.get("label_to_id", {}))
    if label_to_id != GENERATOR_LABEL_TO_ID:
        raise ValueError(f"VAE posterior-bank label map mismatch: {label_to_id}")
    temperature = float(bank_package.get("temperature", -1.0))
    if abs(temperature - VAE_TEMPERATURE) > 1e-8:
        raise ValueError(f"VAE posterior-bank temperature mismatch: {temperature}")
    if bank_package.get("fitted_split") != "train":
        raise ValueError("VAE posterior bank must be fitted on the training split")
    banks = bank_package.get("banks")
    if not isinstance(banks, dict):
        raise ValueError("VAE posterior bank is missing banks")
    output: dict[int, dict[str, Any]] = {}
    for label_id in range(len(GENERATOR_CLASSES)):
        raw = banks.get(label_id, banks.get(str(label_id)))
        if not isinstance(raw, dict):
            raise ValueError(f"VAE posterior bank is missing label {label_id}")
        mu = torch.as_tensor(raw.get("mu"), dtype=torch.float32, device=device)
        logvar = torch.as_tensor(raw.get("logvar"), dtype=torch.float32, device=device)
        paths = list(raw.get("paths", []))
        if mu.shape != (256, 16, 16, 16) or logvar.shape != mu.shape or len(paths) != len(mu):
            raise ValueError(f"VAE posterior bank shape mismatch for label {label_id}")
        output[label_id] = {"mu": mu, "logvar": logvar, "paths": paths}
    return output


def _make_record(
    model: str,
    seed: int,
    label_id: int,
    sample_index: int,
    relative_path: str,
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "species": GENERATOR_CLASSES[label_id],
        "relative_path": relative_path.replace("\\", "/"),
        "generator": model,
        "pool_rank": sample_index,
        "sample_seed": deterministic_sample_seed(seed, label_id, sample_index),
    }
    record.update(extra)
    return record


def _existing_records(output: Path) -> dict[tuple[int, int], dict[str, Any]]:
    manifest = output / "manifest.csv"
    if not manifest.exists():
        return {}
    frame = pd.read_csv(manifest)
    required = {"species", "relative_path", "pool_rank"}
    if not required.issubset(frame.columns):
        raise ValueError(f"pool manifest is missing columns: {sorted(required - set(frame.columns))}")
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        species = str(row["species"])
        if species not in GENERATOR_LABEL_TO_ID:
            raise ValueError(f"pool manifest has unknown species: {species}")
        key = (GENERATOR_LABEL_TO_ID[species], int(row["pool_rank"]))
        records[key] = row
    return records


def _write_metadata(
    output: Path,
    model: str,
    seed: int,
    samples_per_species: int,
    checkpoint: Path,
    posterior_bank: Path | None,
) -> None:
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "generator": model,
        "classes": list(GENERATOR_CLASSES),
        "seed": int(seed),
        "samples_per_class": int(samples_per_species),
        "checkpoint": str(checkpoint),
        "output_scale": "classifier_-1_1",
        "normalization": {"mean": NORMALIZATION_MEAN, "std": NORMALIZATION_STD},
    }
    if model == "vae_v3":
        metadata.update(
            {
                "posterior_bank": str(posterior_bank) if posterior_bank else None,
                "temperature": VAE_TEMPERATURE,
                "sampling_type": "per_species_posterior_anchor_mixture",
            }
        )
    else:
        metadata.update(
            {
                "timesteps": DIFFUSION_TIMESTEPS,
                "beta_schedule": "cosine",
                "sampler": "ddim",
                "ddim_steps": DIFFUSION_DDIM_STEPS,
                "ddim_eta": DIFFUSION_DDIM_ETA,
                "guidance_weight": DIFFUSION_GUIDANCE,
                "clamp_samples": DIFFUSION_CLAMP,
                "ema_state_dict": True,
                "chunk_independent": "per-sample initial noise streams; eta=0",
            }
        )
    _atomic_json_write(metadata, output / "generation.json")


@torch.inference_mode()
def generate_pool(
    model: str,
    checkpoint: Path,
    seed: int,
    samples_per_species: int,
    output: Path,
    device: torch.device,
    posterior_bank: Path | None = None,
    chunk_size: int = 8,
) -> Path:
    """Generate or resume one 200-per-species classifier-input pool."""
    if model not in {"vae_v3", "diffusion"}:
        raise ValueError("model must be vae_v3 or diffusion")
    if seed < 0 or samples_per_species < 1 or chunk_size < 1:
        raise ValueError("seed, samples_per_species, and chunk_size must be positive")
    checkpoint = checkpoint.resolve()
    verify_checkpoint(checkpoint, model)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = _existing_records(output)

    if model == "vae_v3":
        vae, checkpoint_data = load_vae_model(checkpoint, device)
        if checkpoint_parameter_count(vae) != VAE_PARAMETER_COUNT:
            raise ValueError("VAE parameter count mismatch")
        checkpoint_labels = dict(checkpoint_data.get("label_to_id", {}))
        if checkpoint_labels and checkpoint_labels != GENERATOR_LABEL_TO_ID:
            raise ValueError(f"VAE checkpoint label map mismatch: {checkpoint_labels}")
        if posterior_bank is None:
            raise ValueError("vae_v3 requires --posterior-bank")
        bank = _load_posterior_bank(posterior_bank.resolve(), device)
        schedule = None
    else:
        diffusion, checkpoint_data = load_diffusion_model(checkpoint, device)
        if checkpoint_parameter_count(diffusion) != DIFFUSION_PARAMETER_COUNT:
            raise ValueError("diffusion parameter count mismatch")
        checkpoint_labels = dict(checkpoint_data.get("label_to_id", {}))
        if checkpoint_labels and checkpoint_labels != GENERATOR_LABEL_TO_ID:
            raise ValueError(f"diffusion checkpoint label map mismatch: {checkpoint_labels}")
        # The recorded checkpoint stores a DDPM notebook setting.  The
        # evaluation protocol intentionally overrides it with DDIM below.
        schedule = build_diffusion_schedule(device, DIFFUSION_TIMESTEPS)
        vae = None
        bank = None

    all_records: dict[tuple[int, int], dict[str, Any]] = {}
    for label_id, species in enumerate(GENERATOR_CLASSES):
        species_dir = output / "classifier_input" / species_slug(species)
        species_dir.mkdir(parents=True, exist_ok=True)
        pending: list[tuple[int, Path]] = []
        for sample_index in range(samples_per_species):
            path = species_dir / f"{sample_index:04d}.npy"
            key = (label_id, sample_index)
            existing = _valid_classifier_array(path)
            if existing is not None:
                extra = {}
                if model == "vae_v3":
                    extra = {"vae_temperature": VAE_TEMPERATURE}
                else:
                    extra = {
                        "sampler": "ddim",
                        "ddim_steps": DIFFUSION_DDIM_STEPS,
                        "ddim_eta": DIFFUSION_DDIM_ETA,
                        "guidance_weight": DIFFUSION_GUIDANCE,
                        "clamp_samples": DIFFUSION_CLAMP,
                    }
                all_records[key] = _make_record(
                    model,
                    seed,
                    label_id,
                    sample_index,
                    str(path.relative_to(output)),
                    **extra,
                )
            else:
                pending.append((sample_index, path))

        if model == "diffusion" and pending:
            # Fixed internal groups preserve the exact batch shape across
            # interrupted runs.  Existing rows are regenerated only inside an
            # incomplete group and then atomically replaced with the same bytes;
            # a caller's outer --chunk-size therefore cannot affect results.
            work_items = [(sample_index, species_dir / f"{sample_index:04d}.npy") for sample_index in range(samples_per_species)]
            internal_chunk_size = 8
        else:
            work_items = pending
            internal_chunk_size = chunk_size
        for offset in range(0, len(work_items), internal_chunk_size):
            chunk = work_items[offset : offset + internal_chunk_size]
            labels = torch.full((len(chunk),), label_id, dtype=torch.long, device=device)
            samples: torch.Tensor
            extras_per_sample: list[dict[str, Any]] = [{} for _ in chunk]
            if model == "vae_v3":
                assert vae is not None and bank is not None
                latents = []
                for item_index, (sample_index, _) in enumerate(chunk):
                    stream_seed = deterministic_sample_seed(seed, label_id, sample_index)
                    generator = torch.Generator(device=device.type).manual_seed(stream_seed)
                    anchor_index = int(torch.randint(len(bank[label_id]["mu"]), (), generator=generator, device=device))
                    mu = bank[label_id]["mu"][anchor_index]
                    logvar = bank[label_id]["logvar"][anchor_index]
                    latents.append(ConditionalVAE.reparameterize(mu, logvar, generator))
                    extras_per_sample[item_index] = {
                        "vae_anchor_index": anchor_index,
                        "vae_temperature": VAE_TEMPERATURE,
                    }
                samples = vae.decode(torch.stack(latents), labels)
            else:
                assert schedule is not None
                # Every sample gets its own deterministic initial stream, then
                # the fixed internal group is evaluated together.  The U-Net
                # has no batch-normalization, and the fixed group shape makes
                # the result independent of an outer resume/chunk boundary.
                initial_noise = []
                for sample_index, _ in chunk:
                    stream_seed = deterministic_sample_seed(seed, label_id, sample_index)
                    generator = torch.Generator(device=device.type).manual_seed(stream_seed)
                    initial_noise.append(torch.randn((1, 128, 128), generator=generator, device=device))
                samples = ddim_sample(
                    diffusion,
                    labels,
                    torch.Generator(device=device.type).manual_seed(0),
                    schedule,
                    steps=DIFFUSION_DDIM_STEPS,
                    eta=DIFFUSION_DDIM_ETA,
                    guidance=DIFFUSION_GUIDANCE,
                    clamp_samples=DIFFUSION_CLAMP,
                    initial_noise=torch.stack(initial_noise),
                )
                extras_per_sample = [
                    {
                        "sampler": "ddim",
                        "ddim_steps": DIFFUSION_DDIM_STEPS,
                        "ddim_eta": DIFFUSION_DDIM_ETA,
                        "guidance_weight": DIFFUSION_GUIDANCE,
                        "clamp_samples": DIFFUSION_CLAMP,
                    }
                    for _ in chunk
                ]

            classifier_samples = classifier_scale_from_standardized(samples)
            for item_index, (sample_index, path) in enumerate(chunk):
                array = classifier_samples[item_index].detach().cpu().numpy().astype(np.float32)
                temporary = path.with_suffix(path.suffix + ".tmp")
                np.save(temporary, array)
                # numpy appends .npy when the temporary name does not end in
                # .npy; use the actual generated name before replacing.
                generated_temporary = temporary if temporary.exists() else Path(str(temporary) + ".npy")
                generated_temporary.replace(path)
                all_records[(label_id, sample_index)] = _make_record(
                    model,
                    seed,
                    label_id,
                    sample_index,
                    str(path.relative_to(output)),
                    **extras_per_sample[item_index],
                )
            _atomic_dataframe_write(pd.DataFrame(list(all_records.values())), output / "manifest.csv")

    frame = pd.DataFrame(
        [all_records[key] for key in sorted(all_records)],
    )
    expected_rows = len(GENERATOR_CLASSES) * samples_per_species
    if len(frame) != expected_rows:
        raise RuntimeError(f"incomplete generated pool: expected {expected_rows}, got {len(frame)}")
    _atomic_dataframe_write(frame, output / "manifest.csv")
    _write_metadata(output, model, seed, samples_per_species, checkpoint, posterior_bank)
    return output / "manifest.csv"
