from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from bird_song.audio import LogMelTransform, load_waveform
from bird_song.config import DEFAULT_CLASSES, SpectrogramConfig
from bird_song.generation.audio_decode import normalized_logmel_to_waveform


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample held-out WAVs, convert with the current log-mel path, and reconstruct with Griffin-Lim."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "full_dataset_test.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "bird_songs_dataset",
    )
    parser.add_argument(
        "--spectrogram-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "spectrogram.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "current_logmel_reconstruction",
    )
    parser.add_argument("--samples-per-species", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def select_rows(manifest: pd.DataFrame, samples_per_species: int, seed: int) -> list[dict[str, object]]:
    required = {"split", "name", "relative_wav_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    test_rows = manifest[manifest["split"].astype(str).eq("test")]
    if test_rows.empty:
        raise ValueError("The manifest has no held-out test rows")
    if samples_per_species < 1:
        raise ValueError("samples-per-species must be positive")

    rng = np.random.default_rng(seed)
    selected: list[dict[str, object]] = []
    for species in DEFAULT_CLASSES:
        species_rows = test_rows[test_rows["name"].astype(str).eq(species)].reset_index(drop=True)
        if len(species_rows) < samples_per_species:
            raise ValueError(f"Only {len(species_rows)} test rows are available for {species}")
        indices = rng.choice(len(species_rows), size=samples_per_species, replace=False)
        for index in indices:
            selected.append(species_rows.iloc[int(index)].to_dict())
    return selected


def detail_energy(spectrogram: np.ndarray, axis: int) -> float:
    return float(np.abs(np.diff(spectrogram, axis=axis)).mean())


def save_comparison(path: Path, original: np.ndarray, reconstructed: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    difference = np.abs(original - reconstructed)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    panels = (
        (original, "Original log-mel"),
        (reconstructed, "Re-encoded reconstruction"),
        (difference, "Absolute difference"),
    )
    for axis, (panel, title) in zip(axes, panels):
        image = axis.imshow(
            panel,
            origin="lower",
            aspect="auto",
            cmap="magma" if title != "Absolute difference" else "viridis",
            vmin=-1.0 if title != "Absolute difference" else 0.0,
            vmax=1.0,
        )
        axis.set_title(title)
        axis.set_xlabel("Time frame")
        axis.set_ylabel("Mel band")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = SpectrogramConfig.from_json(args.spectrogram_config)
    device = choose_device(args.device)
    manifest = pd.read_csv(args.manifest)
    selected = select_rows(manifest, args.samples_per_species, args.seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transform = LogMelTransform(config, training=False).to(device)
    records: list[dict[str, object]] = []
    torch.manual_seed(args.seed)

    for sample_number, row in enumerate(selected, start=1):
        species = str(row["name"])
        source_path = args.dataset_root / Path(str(row["relative_wav_path"]))
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        sample_dir = args.output_dir / slug(species) / f"sample_{sample_number:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        fixed_waveform = load_waveform(source_path, config, training=False)
        with torch.inference_mode():
            normalized_mel = transform(fixed_waveform.to(device)).squeeze(0)
        if tuple(normalized_mel.shape) != (config.n_mels, config.spectrogram_width):
            raise RuntimeError(f"Unexpected log-mel shape: {tuple(normalized_mel.shape)}")
        normalized_mel_np = normalized_mel.detach().cpu().numpy().astype(np.float32)

        reconstructed = normalized_logmel_to_waveform(
            normalized_mel_np,
            config,
            iterations=args.iterations,
            target_peak=args.target_peak,
            device=device,
        )
        with torch.inference_mode():
            reencoded_mel = transform(torch.from_numpy(reconstructed).to(device)).squeeze(0)
        reencoded_mel_np = reencoded_mel.detach().cpu().numpy().astype(np.float32)

        source_copy = sample_dir / "source_original.wav"
        shutil.copy2(source_path, source_copy)
        sf.write(sample_dir / "input_preprocessed.wav", fixed_waveform.numpy(), config.sample_rate, subtype="PCM_16")
        sf.write(sample_dir / "reconstructed_griffin_lim.wav", reconstructed, config.sample_rate, subtype="PCM_16")
        np.save(sample_dir / "input_logmel.npy", normalized_mel_np)
        np.save(sample_dir / "reencoded_logmel.npy", reencoded_mel_np)
        save_comparison(sample_dir / "logmel_comparison.png", normalized_mel_np, reencoded_mel_np)

        records.append(
            {
                "sample_number": sample_number,
                "species": species,
                "source_file": str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "relative_output_dir": str(sample_dir.relative_to(args.output_dir)).replace("\\", "/"),
                "sample_rate": config.sample_rate,
                "num_samples": int(reconstructed.shape[0]),
                "device": str(device),
                "input_mel_min": float(normalized_mel_np.min()),
                "input_mel_max": float(normalized_mel_np.max()),
                "reencoded_mel_min": float(reencoded_mel_np.min()),
                "reencoded_mel_max": float(reencoded_mel_np.max()),
                "mel_mae": float(np.abs(normalized_mel_np - reencoded_mel_np).mean()),
                "mel_rmse": float(np.sqrt(np.square(normalized_mel_np - reencoded_mel_np).mean())),
                "time_detail_ratio": detail_energy(reencoded_mel_np, axis=1)
                / max(detail_energy(normalized_mel_np, axis=1), 1e-8),
                "frequency_detail_ratio": detail_energy(reencoded_mel_np, axis=0)
                / max(detail_energy(normalized_mel_np, axis=0), 1e-8),
                "input_rms": float(torch.from_numpy(fixed_waveform.numpy()).square().mean().sqrt()),
                "reconstructed_rms": float(np.square(reconstructed).mean() ** 0.5),
                "reconstructed_peak": float(np.max(np.abs(reconstructed))),
            }
        )

    output_manifest = args.output_dir / "reconstruction_manifest.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "description": "Held-out real WAV -> current normalized 128x128 log-mel -> Griffin-Lim reconstruction",
        "seed": args.seed,
        "samples_per_species": args.samples_per_species,
        "griffin_lim_iterations": args.iterations,
        "target_peak": args.target_peak,
        "device": str(device),
        "config": config.to_dict(),
        "source_manifest": str(args.manifest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} reconstructions to {args.output_dir}")
    print(f"Device: {device}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
