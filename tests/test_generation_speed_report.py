from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from bird_song.generation.speed_report import package_speed_report, validate_matched_benchmarks
from bird_song.generation.checkpoint_models import VAE_REPARAMETERIZATION
from bird_song.generation.posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    POSTERIOR_BANK_EXPECTED_COUNTS,
)


def benchmark(model: str, seconds: tuple[float, ...]) -> dict:
    count = 600
    rows = [
        {
            "repeat": index + 1,
            "sample_count": count,
            "seconds": value,
            "samples_per_second": count / value,
            "peak_cuda_memory_bytes": 2_000,
            "incremental_peak_cuda_memory_bytes": 1_000,
        }
        for index, value in enumerate(seconds)
    ]
    throughputs = [row["samples_per_second"] for row in rows]
    return {
        "schema_version": 1,
        "benchmark_type": "generator_only_cuda_synchronized",
        "generator": model,
        "seed": 20260812,
        "samples_per_species": 200,
        "samples_per_repeat": count,
        "classes": ["Northern Cardinal", "Song Sparrow", "American Robin"],
        "balanced_by_species": True,
        "fresh_in_memory_generation_each_repeat": True,
        "existing_arrays_used": False,
        "warmup_batches": 5,
        "repeat_count": 5,
        "batch_size": 8,
        "precision": "float32",
        "cuda_synchronized": True,
        "timing_boundary": "generator only",
        "hardware": {
            "device": "cuda",
            "torch_version": "test",
            "cuda_runtime": "test",
            "cuda_device_index": 0,
            "cuda_device_name": "test GPU",
            "cuda_capability": [8, 9],
            "cuda_total_memory_bytes": 10_000,
        },
        "settings": (
            {
                "sampling_type": "per_species_posterior_anchor_mixture",
                "temperature": 0.35,
                "reparameterization": VAE_REPARAMETERIZATION,
                "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
                "posterior_bank_counts": POSTERIOR_BANK_EXPECTED_COUNTS,
                "posterior_bank_inventory": {
                    species: {
                        "count": POSTERIOR_BANK_EXPECTED_COUNTS[species],
                        "source_indices": list(
                            range(POSTERIOR_BANK_EXPECTED_COUNTS[species])
                        ),
                        "relative_wav_paths": [
                            f"wavfiles/{label_id}-{index}.wav"
                            for index in range(POSTERIOR_BANK_EXPECTED_COUNTS[species])
                        ],
                    }
                    for label_id, species in enumerate(
                        ("Northern Cardinal", "Song Sparrow", "American Robin")
                    )
                },
                "vae_checkpoint_retrained": False,
            }
            if model == "vae_v3"
            else {
                "sampler": "ddim",
                "timesteps": 1000,
                "ddim_steps": 100,
                "ddim_eta": 0.0,
                "guidance_weight": 3.0,
                "clamp_samples": 4.0,
                "beta_schedule": "cosine",
                "ema_state_dict": True,
                "checkpoint_epoch": 34,
                "checkpoint_best_validation_loss": 0.14470337276743556,
                "checkpoint_selection": "validation_best",
                "stored_sampler_overridden": "ddpm",
            }
        ),
        "repeat_results": rows,
        "aggregate": {
            "seconds_mean": sum(seconds) / len(seconds),
            "seconds_sample_sd": statistics.stdev(seconds),
            "seconds_min": min(seconds),
            "seconds_max": max(seconds),
            "samples_per_second_mean": sum(row["samples_per_second"] for row in rows) / len(rows),
            "samples_per_second_sample_sd": statistics.stdev(throughputs),
            "peak_cuda_memory_bytes_max": 2_000,
            "incremental_peak_cuda_memory_bytes_max": 1_000,
        },
    }


def test_validate_rejects_comparison_mismatch() -> None:
    vae = benchmark("vae_v3", (1.0, 1.1, 1.2, 1.3, 1.4))
    diffusion = benchmark("diffusion", (5.0, 5.1, 5.2, 5.3, 5.4))
    diffusion["batch_size"] = 4
    with pytest.raises(ValueError, match="batch_size"):
        validate_matched_benchmarks(vae, diffusion)


def test_validate_rejects_memory_aggregate_mismatch() -> None:
    vae = benchmark("vae_v3", (1.0, 1.1, 1.2, 1.3, 1.4))
    diffusion = benchmark("diffusion", (5.0, 5.1, 5.2, 5.3, 5.4))
    vae["aggregate"]["peak_cuda_memory_bytes_max"] = 1_999
    with pytest.raises(ValueError, match="peak_cuda_memory_bytes_max"):
        validate_matched_benchmarks(vae, diffusion)


def test_validate_rejects_filtered_bank_inventory_mismatch() -> None:
    vae = benchmark("vae_v3", (1.0, 1.1, 1.2, 1.3, 1.4))
    diffusion = benchmark("diffusion", (5.0, 5.1, 5.2, 5.3, 5.4))
    vae["settings"]["posterior_bank_inventory"]["Song Sparrow"]["count"] = 256
    with pytest.raises(ValueError, match="inventory is invalid for Song Sparrow"):
        validate_matched_benchmarks(vae, diffusion)


def test_package_writes_raw_and_derived_evidence(tmp_path: Path) -> None:
    vae_path = tmp_path / "vae.json"
    diffusion_path = tmp_path / "diffusion.json"
    vae_path.write_text(
        json.dumps(benchmark("vae_v3", (1.0, 1.1, 1.2, 1.3, 1.4))),
        encoding="utf-8",
    )
    diffusion_path.write_text(
        json.dumps(benchmark("diffusion", (5.0, 5.1, 5.2, 5.3, 5.4))),
        encoding="utf-8",
    )
    report = tmp_path / "report"

    summary = package_speed_report(vae_path, diffusion_path, report)

    assert summary["comparison"]["diffusion_to_vae_time_ratio"] == pytest.approx(5.2 / 1.2)
    assert (report / "repeat_results.csv").is_file()
    assert (report / "vae_v3_benchmark.json").is_file()
    assert "generator-only" in (report / "README.md").read_text(encoding="utf-8")
