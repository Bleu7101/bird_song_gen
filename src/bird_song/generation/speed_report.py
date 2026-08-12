"""Validation and packaging for matched generator-only speed benchmarks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .checkpoint_models import (
    DIFFUSION_CLAMP,
    DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
    DIFFUSION_CHECKPOINT_EPOCH,
    DIFFUSION_DDIM_ETA,
    DIFFUSION_DDIM_STEPS,
    DIFFUSION_GUIDANCE,
    DIFFUSION_TIMESTEPS,
    DIFFUSION_STORED_SAMPLER,
    GENERATOR_CLASSES,
    VAE_REPARAMETERIZATION,
    VAE_TEMPERATURE,
)
from .posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    POSTERIOR_BANK_EXPECTED_COUNTS,
)


EXPECTED_MODELS = ("vae_v3", "diffusion")
EXPECTED_SAMPLES_PER_SPECIES = 200
EXPECTED_SAMPLES_PER_REPEAT = len(GENERATOR_CLASSES) * EXPECTED_SAMPLES_PER_SPECIES
EXPECTED_BATCH_SIZE = 8
EXPECTED_WARMUP_BATCHES = 5
EXPECTED_REPEAT_COUNT = 5
EXPECTED_PRECISION = "float32"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark must be a JSON object: {path}")
    return value


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _validate_one(payload: Mapping[str, Any], expected_model: str) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark schema for {expected_model}")
    if payload.get("benchmark_type") != "generator_only_cuda_synchronized":
        raise ValueError(f"{expected_model} is not a generator-only synchronized benchmark")
    if payload.get("generator") != expected_model:
        raise ValueError(f"Expected {expected_model}, found {payload.get('generator')}")
    if payload.get("existing_arrays_used") is not False:
        raise ValueError(f"{expected_model} benchmark reused generated arrays")
    if payload.get("fresh_in_memory_generation_each_repeat") is not True:
        raise ValueError(f"{expected_model} benchmark did not use fresh in-memory generation")
    if payload.get("cuda_synchronized") is not True:
        raise ValueError(f"{expected_model} benchmark is not CUDA synchronized")
    expected_contract = {
        "classes": list(GENERATOR_CLASSES),
        "samples_per_species": EXPECTED_SAMPLES_PER_SPECIES,
        "samples_per_repeat": EXPECTED_SAMPLES_PER_REPEAT,
        "balanced_by_species": True,
        "batch_size": EXPECTED_BATCH_SIZE,
        "warmup_batches": EXPECTED_WARMUP_BATCHES,
        "repeat_count": EXPECTED_REPEAT_COUNT,
        "precision": EXPECTED_PRECISION,
    }
    for field, expected in expected_contract.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"{expected_model} benchmark has noncanonical {field}: "
                f"{payload.get(field)!r} != {expected!r}"
            )
    repeats = payload.get("repeat_results")
    if not isinstance(repeats, list):
        raise ValueError(f"{expected_model} benchmark repeat rows are missing")
    if len(repeats) != int(payload.get("repeat_count", -1)):
        raise ValueError(f"{expected_model} repeat count does not match its rows")
    expected_samples = int(payload["samples_per_repeat"])
    seconds_values: list[float] = []
    throughput_values: list[float] = []
    for row in repeats:
        if int(row.get("sample_count", -1)) != expected_samples:
            raise ValueError(f"{expected_model} repeat has a mismatched sample count")
        seconds = float(row.get("seconds", 0.0))
        throughput = float(row.get("samples_per_second", 0.0))
        if seconds <= 0 or throughput <= 0:
            raise ValueError(f"{expected_model} repeat has a non-positive timing value")
        if not math.isclose(throughput, expected_samples / seconds, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{expected_model} repeat throughput does not match count / seconds")
        seconds_values.append(seconds)
        throughput_values.append(throughput)

    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"{expected_model} benchmark aggregate is missing")
    seconds_array = np.asarray(seconds_values, dtype=np.float64)
    throughput_array = np.asarray(throughput_values, dtype=np.float64)
    recomputed = {
        "seconds_mean": float(seconds_array.mean()),
        "seconds_sample_sd": float(seconds_array.std(ddof=1)),
        "seconds_min": float(seconds_array.min()),
        "seconds_max": float(seconds_array.max()),
        "samples_per_second_mean": float(throughput_array.mean()),
        "samples_per_second_sample_sd": float(throughput_array.std(ddof=1)),
    }
    for field, expected in recomputed.items():
        observed = float(aggregate.get(field, float("nan")))
        if not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(
                f"{expected_model} benchmark aggregate does not match repeat rows for {field}"
            )
    memory_maxima = {
        "peak_cuda_memory_bytes_max": max(
            int(row.get("peak_cuda_memory_bytes", -1)) for row in repeats
        ),
        "incremental_peak_cuda_memory_bytes_max": max(
            int(row.get("incremental_peak_cuda_memory_bytes", -1)) for row in repeats
        ),
    }
    for field, expected in memory_maxima.items():
        if expected < 0 or int(aggregate.get(field, -1)) != expected:
            raise ValueError(
                f"{expected_model} benchmark aggregate does not match repeat rows for {field}"
            )

    settings = payload.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError(f"{expected_model} benchmark settings are missing")
    if expected_model == "vae_v3":
        if (
            settings.get("sampling_type") != "per_species_posterior_anchor_mixture"
            or not math.isclose(
                float(settings.get("temperature", float("nan"))),
                VAE_TEMPERATURE,
                abs_tol=1e-12,
            )
            or settings.get("reparameterization") != VAE_REPARAMETERIZATION
            or settings.get("posterior_bank_contract") != POSTERIOR_BANK_CONTRACT
            or settings.get("posterior_bank_counts") != POSTERIOR_BANK_EXPECTED_COUNTS
            or settings.get("vae_checkpoint_retrained") is not False
        ):
            raise ValueError(
                "VAE benchmark does not use the corrected filtered-bank sampling contract"
            )
        inventory = settings.get("posterior_bank_inventory")
        if not isinstance(inventory, Mapping):
            raise ValueError("VAE benchmark is missing the filtered posterior-bank inventory")
        for species in GENERATOR_CLASSES:
            entry = inventory.get(species)
            expected_count = POSTERIOR_BANK_EXPECTED_COUNTS[species]
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"VAE benchmark is missing posterior-bank inventory for {species}"
                )
            source_indices = entry.get("source_indices")
            relative_wav_paths = entry.get("relative_wav_paths")
            try:
                recorded_count = int(entry.get("count", -1))
                numeric_indices = [float(value) for value in source_indices]
            except (TypeError, ValueError, OverflowError):
                recorded_count = -1
                numeric_indices = []
            normalized_paths = (
                [str(value).replace("\\", "/") for value in relative_wav_paths]
                if isinstance(relative_wav_paths, list)
                else []
            )
            if (
                recorded_count != expected_count
                or not isinstance(source_indices, list)
                or len(numeric_indices) != expected_count
                or any(
                    not math.isfinite(value) or not value.is_integer() or value < 0
                    for value in numeric_indices
                )
                or len({int(value) for value in numeric_indices}) != expected_count
                or len(normalized_paths) != expected_count
                or len(set(normalized_paths)) != expected_count
                or any(not path.startswith("wavfiles/") for path in normalized_paths)
            ):
                raise ValueError(
                    f"VAE benchmark posterior-bank inventory is invalid for {species}"
                )
        return

    diffusion_contract = {
        "sampler": "ddim",
        "timesteps": DIFFUSION_TIMESTEPS,
        "ddim_steps": DIFFUSION_DDIM_STEPS,
        "ddim_eta": DIFFUSION_DDIM_ETA,
        "guidance_weight": DIFFUSION_GUIDANCE,
        "clamp_samples": DIFFUSION_CLAMP,
        "beta_schedule": "cosine",
        "ema_state_dict": True,
        "checkpoint_epoch": DIFFUSION_CHECKPOINT_EPOCH,
        "checkpoint_best_validation_loss": DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
        "checkpoint_selection": "validation_best",
        "stored_sampler_overridden": DIFFUSION_STORED_SAMPLER,
    }
    for field, expected in diffusion_contract.items():
        if settings.get(field) != expected:
            raise ValueError(
                f"Diffusion benchmark has noncanonical {field}: "
                f"{settings.get(field)!r} != {expected!r}"
            )


def validate_matched_benchmarks(
    vae: Mapping[str, Any],
    diffusion: Mapping[str, Any],
) -> None:
    """Require the comparison dimensions that materially affect generator speed."""
    _validate_one(vae, "vae_v3")
    _validate_one(diffusion, "diffusion")
    exact_fields = (
        "seed",
        "samples_per_species",
        "samples_per_repeat",
        "classes",
        "balanced_by_species",
        "batch_size",
        "precision",
        "warmup_batches",
        "repeat_count",
        "timing_boundary",
    )
    for field in exact_fields:
        if vae.get(field) != diffusion.get(field):
            raise ValueError(f"Benchmark mismatch for {field}: {vae.get(field)!r} != {diffusion.get(field)!r}")
    hardware_fields = (
        "device",
        "torch_version",
        "cuda_runtime",
        "cuda_device_index",
        "cuda_device_name",
        "cuda_capability",
        "cuda_total_memory_bytes",
    )
    vae_hardware = vae.get("hardware", {})
    diffusion_hardware = diffusion.get("hardware", {})
    for field in hardware_fields:
        if vae_hardware.get(field) != diffusion_hardware.get(field):
            raise ValueError(
                f"Hardware mismatch for {field}: {vae_hardware.get(field)!r} != {diffusion_hardware.get(field)!r}"
            )


def package_speed_report(
    vae_benchmark: Path,
    diffusion_benchmark: Path,
    report_dir: Path,
    *,
    measurement_sequence_note: str | None = None,
) -> dict[str, Any]:
    vae = _load_json(vae_benchmark.resolve())
    diffusion = _load_json(diffusion_benchmark.resolve())
    validate_matched_benchmarks(vae, diffusion)

    payloads = {"vae_v3": vae, "diffusion": diffusion}
    repeat_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        payload = payloads[model]
        aggregate = dict(payload["aggregate"])
        seconds_values = np.asarray(
            [float(row["seconds"]) for row in payload["repeat_results"]],
            dtype=np.float64,
        )
        throughput_values = np.asarray(
            [float(row["samples_per_second"]) for row in payload["repeat_results"]],
            dtype=np.float64,
        )
        models[model] = {
            "seconds_mean": float(seconds_values.mean()),
            "seconds_sample_sd": float(seconds_values.std(ddof=1)),
            "seconds_median": float(np.median(seconds_values)),
            "seconds_q1": float(np.quantile(seconds_values, 0.25)),
            "seconds_q3": float(np.quantile(seconds_values, 0.75)),
            "seconds_min": float(seconds_values.min()),
            "seconds_max": float(seconds_values.max()),
            "samples_per_second_mean": float(throughput_values.mean()),
            "samples_per_second_sample_sd": float(throughput_values.std(ddof=1)),
            "samples_per_second_median": float(np.median(throughput_values)),
            "peak_cuda_memory_bytes_max": int(aggregate["peak_cuda_memory_bytes_max"]),
            "incremental_peak_cuda_memory_bytes_max": int(
                aggregate["incremental_peak_cuda_memory_bytes_max"]
            ),
        }
        for row in payload["repeat_results"]:
            repeat_rows.append({"model": model, **row})

    vae_seconds = models["vae_v3"]["seconds_mean"]
    diffusion_seconds = models["diffusion"]["seconds_mean"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "title": "Matched VAE-v3 and diffusion generator-only speed comparison",
        "sample_count_per_repeat": int(vae["samples_per_repeat"]),
        "samples_per_species": int(vae["samples_per_species"]),
        "repeat_count_per_model": int(vae["repeat_count"]),
        "warmup_batches_per_model": int(vae["warmup_batches"]),
        "batch_size": int(vae["batch_size"]),
        "precision": str(vae["precision"]),
        "cuda_synchronized": True,
        "timing_boundary": str(vae["timing_boundary"]),
        "hardware": dict(vae["hardware"]),
        "models": models,
        "comparison": {
            "absolute_seconds_difference": diffusion_seconds - vae_seconds,
            "diffusion_to_vae_time_ratio": diffusion_seconds / vae_seconds,
            "vae_to_diffusion_throughput_ratio": (
                models["vae_v3"]["samples_per_second_mean"]
                / models["diffusion"]["samples_per_second_mean"]
            ),
        },
        "interpretation_boundary": [
            "generator-only spectrogram sampling on the recorded hardware",
            "excludes checkpoint loading, conversion to classifier scale, CPU transfer, file I/O, and waveform decoding",
            "runtime values are system measurements, not model-quality metrics",
            "models were benchmarked sequentially rather than interleaved",
            "repeats reuse the same deterministic sample streams, so timing variation is system variation",
        ],
    }
    if measurement_sequence_note:
        summary["measurement_sequence_note"] = measurement_sequence_note

    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(vae, report_dir / "vae_v3_benchmark.json")
    _atomic_json(diffusion, report_dir / "diffusion_benchmark.json")
    _atomic_json(summary, report_dir / "summary.json")
    _atomic_csv(pd.DataFrame(repeat_rows), report_dir / "repeat_results.csv")

    vae_sd = models["vae_v3"]["seconds_sample_sd"]
    diffusion_sd = models["diffusion"]["seconds_sample_sd"]
    readme = f"""# Generator-only speed comparison

This matched benchmark compares fresh in-memory VAE-v3 and diffusion generation
for {summary['sample_count_per_repeat']} spectrograms per repeat
({summary['samples_per_species']} per species). Both models use FP32, batch size
{summary['batch_size']}, the same CUDA device, {summary['warmup_batches_per_model']}
warm-up batches, and {summary['repeat_count_per_model']} synchronized repeats.

| Model | Mean seconds / {summary['sample_count_per_repeat']} | Sample SD | Median [Q1, Q3] seconds | Mean spectrograms/s | Peak CUDA memory |
|---|---:|---:|---:|---:|---:|
| VAE-v3 | {vae_seconds:.4f} | {vae_sd:.4f} | {models['vae_v3']['seconds_median']:.4f} [{models['vae_v3']['seconds_q1']:.4f}, {models['vae_v3']['seconds_q3']:.4f}] | {models['vae_v3']['samples_per_second_mean']:.3f} | {models['vae_v3']['peak_cuda_memory_bytes_max'] / 2**30:.3f} GiB |
| Diffusion | {diffusion_seconds:.4f} | {diffusion_sd:.4f} | {models['diffusion']['seconds_median']:.4f} [{models['diffusion']['seconds_q1']:.4f}, {models['diffusion']['seconds_q3']:.4f}] | {models['diffusion']['samples_per_second_mean']:.3f} | {models['diffusion']['peak_cuda_memory_bytes_max'] / 2**30:.3f} GiB |

Diffusion took **{summary['comparison']['absolute_seconds_difference']:.4f} seconds
longer** per equal-count request and **{summary['comparison']['diffusion_to_vae_time_ratio']:.2f}x**
the VAE-v3 generator time in this environment.

## Boundary

This is generator-only spectrogram latency. It includes posterior-anchor and
latent sampling plus VAE decoding, or initial-noise construction plus 100-step
DDIM sampling. It excludes checkpoint and posterior-bank loading, warm-up,
classifier-scale conversion, CPU transfer, array serialization, report I/O,
waveform decoding, and WAV writing. Runtime is reported separately from quality
and downstream CRNN utility.

The models were benchmarked sequentially rather than interleaved. Each repeat
reuses the same deterministic sample streams, so its variation measures system
runtime variation rather than variation across generated sample populations.
{f"\nMeasurement sequence note: {measurement_sequence_note}\n" if measurement_sequence_note else ""}

Raw repeat-level measurements are in `repeat_results.csv`; the complete captured
environments and model settings are in `vae_v3_benchmark.json` and
`diffusion_benchmark.json`.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary
