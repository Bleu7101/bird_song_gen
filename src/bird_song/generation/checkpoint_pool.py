"""Resumable, deterministic classifier-input pools for frozen generator checkpoints."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .checkpoint_models import (
    DIFFUSION_CLAMP,
    DIFFUSION_DDIM_ETA,
    DIFFUSION_DDIM_STEPS,
    DIFFUSION_GUIDANCE,
    DIFFUSION_STORED_SAMPLER,
    DIFFUSION_PARAMETER_COUNT,
    DIFFUSION_TIMESTEPS,
    GENERATOR_CLASSES,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    VAE_PARAMETER_COUNT,
    VAE_REPARAMETERIZATION,
    VAE_TEMPERATURE,
    ConditionalVAE,
    build_diffusion_schedule,
    checkpoint_parameter_count,
    classifier_scale_from_standardized,
    ddim_sample,
    diffusion_checkpoint_selection,
    load_diffusion_model,
    load_vae_model,
)
from .posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    POSTERIOR_BANK_EXPECTED_COUNTS,
    POSTERIOR_BANK_SCHEMA_VERSION,
    POSTERIOR_BANK_SOURCE_MANIFEST,
)


GENERATOR_LABEL_TO_ID = {name: index for index, name in enumerate(GENERATOR_CLASSES)}
GENERATOR_ONLY_TIMING_BOUNDARY = (
    "fresh in-memory generator sampling only; includes VAE posterior-anchor selection, latent sampling, "
    "and decoder forward or diffusion initial-noise construction and DDIM sampling; excludes checkpoint "
    "and posterior-bank loading, warm-up, classifier-scale conversion, CPU transfer, array serialization, "
    "and manifest/metadata I/O"
)


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
    if int(bank_package.get("schema_version", -1)) != POSTERIOR_BANK_SCHEMA_VERSION:
        raise ValueError("VAE posterior bank does not use the filtered-bank schema")
    if bank_package.get("posterior_bank_contract") != POSTERIOR_BANK_CONTRACT:
        raise ValueError("VAE posterior bank does not use the content-safe filtered-bank contract")
    if bank_package.get("source_manifest") != POSTERIOR_BANK_SOURCE_MANIFEST:
        raise ValueError("VAE posterior-bank source manifest mismatch")
    if bank_package.get("derivation") != "filtered_existing_posterior_bank":
        raise ValueError("VAE posterior bank must be derived by filtering the existing bank")
    if bank_package.get("vae_checkpoint_retrained") is not False:
        raise ValueError("VAE posterior bank must record that the checkpoint was not retrained")
    if dict(bank_package.get("counts", {})) != POSTERIOR_BANK_EXPECTED_COUNTS:
        raise ValueError("VAE posterior-bank class counts mismatch")
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
        paths = [str(path).replace("\\", "/") for path in raw.get("relative_wav_paths", [])]
        source_indices = [int(index) for index in raw.get("source_indices", [])]
        if (
            mu.ndim != 4
            or tuple(mu.shape[1:]) != (16, 16, 16)
            or len(mu) == 0
            or logvar.shape != mu.shape
            or len(paths) != len(mu)
            or len(source_indices) != len(mu)
            or len(set(source_indices)) != len(source_indices)
            or any(not path.startswith("wavfiles/") for path in paths)
        ):
            raise ValueError(f"VAE posterior bank shape mismatch for label {label_id}")
        species = GENERATOR_CLASSES[label_id]
        if len(mu) != POSTERIOR_BANK_EXPECTED_COUNTS[species]:
            raise ValueError(f"VAE posterior-bank count mismatch for {species}")
        output[label_id] = {
            "mu": mu,
            "logvar": logvar,
            "source_indices": source_indices,
            "relative_wav_paths": paths,
        }
    return output


def _posterior_bank_inventory(bank: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        GENERATOR_CLASSES[label_id]: {
            "count": len(bank[label_id]["mu"]),
            "source_indices": list(bank[label_id]["source_indices"]),
            "relative_wav_paths": list(bank[label_id]["relative_wav_paths"]),
        }
        for label_id in range(len(GENERATOR_CLASSES))
    }


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


def _record_matches_sampling_contract(
    record: Mapping[str, Any] | None,
    model: str,
    seed: int,
    label_id: int,
    sample_index: int,
) -> bool:
    """Return whether an existing array was generated under the current sampler contract."""
    if record is None:
        return False
    try:
        if str(record.get("generator")) != model:
            return False
        if str(record.get("species")) != GENERATOR_CLASSES[label_id]:
            return False
        if int(record.get("pool_rank", -1)) != sample_index:
            return False
        if int(record.get("sample_seed", -1)) != deterministic_sample_seed(seed, label_id, sample_index):
            return False
        if model == "vae_v3":
            return (
                abs(float(record.get("vae_temperature", -1.0)) - VAE_TEMPERATURE) <= 1e-8
                and str(record.get("vae_reparameterization", "")) == VAE_REPARAMETERIZATION
                and int(record.get("vae_anchor_index", -1)) >= 0
                and str(record.get("vae_posterior_bank_contract", "")) == POSTERIOR_BANK_CONTRACT
                and int(record.get("vae_bank_class_count", -1))
                == POSTERIOR_BANK_EXPECTED_COUNTS[GENERATOR_CLASSES[label_id]]
                and int(record.get("vae_anchor_source_index", -1)) >= 0
                and str(record.get("vae_anchor_relative_wav_path", "")).startswith("wavfiles/")
            )
        return (
            str(record.get("sampler", "")).lower() == "ddim"
            and int(record.get("ddim_steps", -1)) == DIFFUSION_DDIM_STEPS
            and abs(float(record.get("ddim_eta", -1.0)) - DIFFUSION_DDIM_ETA) <= 1e-8
            and abs(float(record.get("guidance_weight", float("nan"))) - DIFFUSION_GUIDANCE) <= 1e-8
            and abs(float(record.get("clamp_samples", float("nan"))) - DIFFUSION_CLAMP) <= 1e-8
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _reused_record_extras(record: Mapping[str, Any], model: str) -> dict[str, Any]:
    if model == "vae_v3":
        return {
            "vae_anchor_index": int(record["vae_anchor_index"]),
            "vae_temperature": VAE_TEMPERATURE,
            "vae_reparameterization": VAE_REPARAMETERIZATION,
            "vae_posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
            "vae_bank_class_count": int(record["vae_bank_class_count"]),
            "vae_anchor_source_index": int(record["vae_anchor_source_index"]),
            "vae_anchor_relative_wav_path": str(record["vae_anchor_relative_wav_path"]),
        }
    return {
        "sampler": "ddim",
        "ddim_steps": DIFFUSION_DDIM_STEPS,
        "ddim_eta": DIFFUSION_DDIM_ETA,
        "guidance_weight": DIFFUSION_GUIDANCE,
        "clamp_samples": DIFFUSION_CLAMP,
    }


def _write_metadata(
    output: Path,
    model: str,
    seed: int,
    samples_per_species: int,
    checkpoint: Path,
    posterior_bank: Path | None,
    generation_batch_size: int,
    checkpoint_data: Mapping[str, Any] | None = None,
    bank: Mapping[int, Mapping[str, Any]] | None = None,
) -> None:
    metadata: dict[str, Any] = {
        "schema_version": 3,
        "generator": model,
        "classes": list(GENERATOR_CLASSES),
        "seed": int(seed),
        "samples_per_class": int(samples_per_species),
        "checkpoint": str(checkpoint),
        "output_scale": "classifier_-1_1",
        "normalization": {"mean": NORMALIZATION_MEAN, "std": NORMALIZATION_STD},
        "generation_batch_size": int(generation_batch_size),
    }
    if model == "vae_v3":
        if bank is None:
            raise ValueError("VAE generation metadata requires the filtered posterior bank")
        metadata.update(
            {
                "posterior_bank": str(posterior_bank) if posterior_bank else None,
                "temperature": VAE_TEMPERATURE,
                "reparameterization": VAE_REPARAMETERIZATION,
                "sampling_type": "per_species_posterior_anchor_mixture",
                "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                "posterior_bank_source_manifest": POSTERIOR_BANK_SOURCE_MANIFEST,
                "posterior_bank_derivation": "filtered_existing_posterior_bank",
                "vae_checkpoint_retrained": False,
                "posterior_bank_counts": dict(POSTERIOR_BANK_EXPECTED_COUNTS),
                "posterior_bank_inventory": _posterior_bank_inventory(bank),
            }
        )
    else:
        if checkpoint_data is None:
            raise ValueError("diffusion generation metadata requires checkpoint selection details")
        selection = diffusion_checkpoint_selection(checkpoint_data)
        stored_sampler = str(
            dict(checkpoint_data.get("config", {})).get("SAMPLER", "")
        ).lower()
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
                **selection,
                "stored_sampler_overridden": stored_sampler,
                "chunk_independent": "per-sample initial noise streams; eta=0",
            }
        )
    _atomic_json_write(metadata, output / "generation.json")


def _prepare_generator(
    model: str,
    checkpoint: Path,
    device: torch.device,
    posterior_bank: Path | None,
) -> tuple[
    torch.nn.Module,
    dict[int, dict[str, Any]] | None,
    dict[str, torch.Tensor] | None,
    dict[str, Any],
]:
    checkpoint = checkpoint.resolve()
    verify_checkpoint(checkpoint, model)
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
        return vae, bank, None, checkpoint_data

    diffusion, checkpoint_data = load_diffusion_model(checkpoint, device)
    if checkpoint_parameter_count(diffusion) != DIFFUSION_PARAMETER_COUNT:
        raise ValueError("diffusion parameter count mismatch")
    checkpoint_labels = dict(checkpoint_data.get("label_to_id", {}))
    if checkpoint_labels and checkpoint_labels != GENERATOR_LABEL_TO_ID:
        raise ValueError(f"diffusion checkpoint label map mismatch: {checkpoint_labels}")
    stored_sampler = str(dict(checkpoint_data.get("config", {})).get("SAMPLER", "")).lower()
    if stored_sampler != DIFFUSION_STORED_SAMPLER:
        raise ValueError(
            f"diffusion checkpoint stored sampler mismatch: {stored_sampler!r}"
        )
    # The recorded checkpoint stores a DDPM notebook setting. The portable
    # evaluation and benchmark contracts intentionally use DDIM.
    schedule = build_diffusion_schedule(device, DIFFUSION_TIMESTEPS)
    return diffusion, None, schedule, checkpoint_data


def _sample_standardized_chunk(
    model: str,
    generator_model: torch.nn.Module,
    bank: dict[int, dict[str, Any]] | None,
    schedule: Mapping[str, torch.Tensor] | None,
    seed: int,
    label_id: int,
    sample_indices: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")
    labels = torch.full((len(sample_indices),), label_id, dtype=torch.long, device=device)
    if model == "vae_v3":
        if bank is None or not isinstance(generator_model, ConditionalVAE):
            raise ValueError("VAE generation state is incomplete")
        latents = []
        extras: list[dict[str, Any]] = []
        for sample_index in sample_indices:
            stream_seed = deterministic_sample_seed(seed, label_id, sample_index)
            random_generator = torch.Generator(device=device.type).manual_seed(stream_seed)
            anchor_index = int(
                torch.randint(
                    len(bank[label_id]["mu"]),
                    (),
                    generator=random_generator,
                    device=device,
                )
            )
            mu = bank[label_id]["mu"][anchor_index]
            logvar = bank[label_id]["logvar"][anchor_index]
            latents.append(
                ConditionalVAE.reparameterize(
                    mu,
                    logvar,
                    random_generator,
                    temperature=VAE_TEMPERATURE,
                )
            )
            extras.append(
                {
                    "vae_anchor_index": anchor_index,
                    "vae_temperature": VAE_TEMPERATURE,
                    "vae_reparameterization": VAE_REPARAMETERIZATION,
                    "vae_posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                    "vae_bank_class_count": len(bank[label_id]["mu"]),
                    "vae_anchor_source_index": int(bank[label_id]["source_indices"][anchor_index]),
                    "vae_anchor_relative_wav_path": str(
                        bank[label_id]["relative_wav_paths"][anchor_index]
                    ),
                }
            )
        return generator_model.decode(torch.stack(latents), labels), extras

    if schedule is None:
        raise ValueError("diffusion generation state is incomplete")
    initial_noise = []
    for sample_index in sample_indices:
        stream_seed = deterministic_sample_seed(seed, label_id, sample_index)
        random_generator = torch.Generator(device=device.type).manual_seed(stream_seed)
        initial_noise.append(torch.randn((1, 128, 128), generator=random_generator, device=device))
    samples = ddim_sample(
        generator_model,
        labels,
        torch.Generator(device=device.type).manual_seed(0),
        schedule,
        steps=DIFFUSION_DDIM_STEPS,
        eta=DIFFUSION_DDIM_ETA,
        guidance=DIFFUSION_GUIDANCE,
        clamp_samples=DIFFUSION_CLAMP,
        initial_noise=torch.stack(initial_noise),
    )
    extras = [
        {
            "sampler": "ddim",
            "ddim_steps": DIFFUSION_DDIM_STEPS,
            "ddim_eta": DIFFUSION_DDIM_ETA,
            "guidance_weight": DIFFUSION_GUIDANCE,
            "clamp_samples": DIFFUSION_CLAMP,
        }
        for _ in sample_indices
    ]
    return samples, extras


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hardware_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["cuda_device_index"] = int(index)
        metadata["cuda_device_name"] = properties.name
        metadata["cuda_capability"] = [int(properties.major), int(properties.minor)]
        metadata["cuda_total_memory_bytes"] = int(properties.total_memory)
        metadata["cuda_multiprocessor_count"] = int(properties.multi_processor_count)
    return metadata


def _model_precision(model: torch.nn.Module) -> str:
    dtypes = {str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()}
    return next(iter(dtypes)) if len(dtypes) == 1 else "mixed:" + ",".join(sorted(dtypes))


@torch.inference_mode()
def benchmark_generation(
    model: str,
    checkpoint: Path,
    seed: int,
    samples_per_species: int,
    device: torch.device,
    posterior_bank: Path | None = None,
    batch_size: int = 8,
    warmup_batches: int = 5,
    repeats: int = 5,
    metadata_output: Path | None = None,
) -> dict[str, Any]:
    """Benchmark fresh generator sampling without loading, conversion, or file I/O in the timed region."""
    if model not in {"vae_v3", "diffusion"}:
        raise ValueError("model must be vae_v3 or diffusion")
    if seed < 0 or samples_per_species < 1 or batch_size < 1:
        raise ValueError("seed, samples_per_species, and batch_size must be positive")
    if warmup_batches < 1 or repeats < 2:
        raise ValueError("benchmark requires at least one warm-up batch and two timed repeats")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("comparable generator benchmarks require an available CUDA device")

    checkpoint = checkpoint.resolve()
    posterior_bank = posterior_bank.resolve() if posterior_bank is not None else None
    generator_model, bank, schedule, checkpoint_data = _prepare_generator(
        model, checkpoint, device, posterior_bank
    )
    generator_model.eval()
    generator_model.requires_grad_(False)

    for warmup_index in range(warmup_batches):
        label_id = warmup_index % len(GENERATOR_CLASSES)
        start = (warmup_index * batch_size) % samples_per_species
        sample_indices = [(start + offset) % samples_per_species for offset in range(batch_size)]
        samples, _ = _sample_standardized_chunk(
            model,
            generator_model,
            bank,
            schedule,
            seed,
            label_id,
            sample_indices,
            device,
        )
        del samples
    _synchronize_cuda(device)

    repeat_results: list[dict[str, Any]] = []
    total_samples = len(GENERATOR_CLASSES) * samples_per_species
    for repeat_index in range(repeats):
        baseline_memory = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        _synchronize_cuda(device)
        started = time.perf_counter()
        for label_id in range(len(GENERATOR_CLASSES)):
            for offset in range(0, samples_per_species, batch_size):
                sample_indices = list(range(offset, min(offset + batch_size, samples_per_species)))
                samples, _ = _sample_standardized_chunk(
                    model,
                    generator_model,
                    bank,
                    schedule,
                    seed,
                    label_id,
                    sample_indices,
                    device,
                )
                del samples
        _synchronize_cuda(device)
        seconds = time.perf_counter() - started
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        repeat_results.append(
            {
                "repeat": repeat_index + 1,
                "sample_count": total_samples,
                "seconds": seconds,
                "samples_per_second": total_samples / seconds,
                "peak_cuda_memory_bytes": peak_memory,
                "incremental_peak_cuda_memory_bytes": max(0, peak_memory - baseline_memory),
            }
        )

    seconds_values = np.asarray([row["seconds"] for row in repeat_results], dtype=np.float64)
    throughput_values = np.asarray(
        [row["samples_per_second"] for row in repeat_results], dtype=np.float64
    )
    model_settings: dict[str, Any]
    if model == "vae_v3":
        model_settings = {
            "sampling_type": "per_species_posterior_anchor_mixture",
            "posterior_bank": str(posterior_bank),
            "temperature": VAE_TEMPERATURE,
            "reparameterization": VAE_REPARAMETERIZATION,
            "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
            "posterior_bank_counts": dict(POSTERIOR_BANK_EXPECTED_COUNTS),
            "posterior_bank_inventory": _posterior_bank_inventory(bank or {}),
            "vae_checkpoint_retrained": False,
        }
    else:
        selection = diffusion_checkpoint_selection(checkpoint_data)
        stored_sampler = str(
            dict(checkpoint_data.get("config", {})).get("SAMPLER", "")
        ).lower()
        model_settings = {
            "sampler": "ddim",
            "timesteps": DIFFUSION_TIMESTEPS,
            "ddim_steps": DIFFUSION_DDIM_STEPS,
            "ddim_eta": DIFFUSION_DDIM_ETA,
            "guidance_weight": DIFFUSION_GUIDANCE,
            "clamp_samples": DIFFUSION_CLAMP,
            "beta_schedule": "cosine",
            "ema_state_dict": True,
            **selection,
            "stored_sampler_overridden": stored_sampler,
        }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_type": "generator_only_cuda_synchronized",
        "generator": model,
        "checkpoint": str(checkpoint),
        "seed": int(seed),
        "classes": list(GENERATOR_CLASSES),
        "samples_per_species": int(samples_per_species),
        "samples_per_repeat": total_samples,
        "balanced_by_species": True,
        "fresh_in_memory_generation_each_repeat": True,
        "existing_arrays_used": False,
        "warmup_batches": int(warmup_batches),
        "repeat_count": int(repeats),
        "batch_size": int(batch_size),
        "precision": _model_precision(generator_model),
        "cuda_synchronized": True,
        "timing_boundary": GENERATOR_ONLY_TIMING_BOUNDARY,
        "settings": model_settings,
        "hardware": _hardware_metadata(device),
        "repeat_results": repeat_results,
        "aggregate": {
            "seconds_mean": float(seconds_values.mean()),
            "seconds_sample_sd": float(seconds_values.std(ddof=1)),
            "seconds_min": float(seconds_values.min()),
            "seconds_max": float(seconds_values.max()),
            "samples_per_second_mean": float(throughput_values.mean()),
            "samples_per_second_sample_sd": float(throughput_values.std(ddof=1)),
            "peak_cuda_memory_bytes_max": int(
                max(row["peak_cuda_memory_bytes"] for row in repeat_results)
            ),
            "incremental_peak_cuda_memory_bytes_max": int(
                max(row["incremental_peak_cuda_memory_bytes"] for row in repeat_results)
            ),
        },
    }
    if metadata_output is not None:
        _atomic_json_write(metadata, metadata_output.resolve())
    return metadata


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
    posterior_bank = posterior_bank.resolve() if posterior_bank is not None else None
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = _existing_records(output)
    generator_model, bank, schedule, checkpoint_data = _prepare_generator(
        model, checkpoint, device, posterior_bank
    )
    generator_model.eval()
    generator_model.requires_grad_(False)

    all_records: dict[tuple[int, int], dict[str, Any]] = {}
    for label_id, species in enumerate(GENERATOR_CLASSES):
        species_dir = output / "classifier_input" / species_slug(species)
        species_dir.mkdir(parents=True, exist_ok=True)
        pending: list[tuple[int, Path]] = []
        for sample_index in range(samples_per_species):
            path = species_dir / f"{sample_index:04d}.npy"
            key = (label_id, sample_index)
            existing = _valid_classifier_array(path)
            existing_record = records.get(key)
            if existing is not None and _record_matches_sampling_contract(
                existing_record,
                model,
                seed,
                label_id,
                sample_index,
            ):
                assert existing_record is not None
                all_records[key] = _make_record(
                    model,
                    seed,
                    label_id,
                    sample_index,
                    str(path.relative_to(output)),
                    **_reused_record_extras(existing_record, model),
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
            samples, extras_per_sample = _sample_standardized_chunk(
                model,
                generator_model,
                bank,
                schedule,
                seed,
                label_id,
                [sample_index for sample_index, _ in chunk],
                device,
            )

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
    generation_batch_size = 8 if model == "diffusion" else chunk_size
    _write_metadata(
        output,
        model,
        seed,
        samples_per_species,
        checkpoint,
        posterior_bank,
        generation_batch_size,
        checkpoint_data,
        bank,
    )
    return output / "manifest.csv"
