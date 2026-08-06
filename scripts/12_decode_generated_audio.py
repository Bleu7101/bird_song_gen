from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bird_song.config import SpectrogramConfig
from bird_song.generation.audio_decode import normalized_logmel_to_waveform, write_waveform


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode generated normalized log-mel arrays to WAV with Griffin-Lim.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/generated_audio")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-species", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SpectrogramConfig.from_json(args.spectrogram_config)
    manifest = pd.read_csv(args.manifest)
    required = {"species", "relative_path", "sample_index"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    species_counts: dict[str, int] = {}
    for row_index, row in manifest.iterrows():
        if args.max_samples is not None and row_index >= args.max_samples:
            break
        species = str(row["species"])
        if args.samples_per_species is not None:
            if args.samples_per_species < 1:
                raise ValueError("samples-per-species must be positive")
            if species_counts.get(species, 0) >= args.samples_per_species:
                continue
        input_path = args.manifest.parent / Path(row["relative_path"])
        spectrogram = np.load(input_path, allow_pickle=False)
        waveform = normalized_logmel_to_waveform(spectrogram, config, args.iterations, args.target_peak)
        destination = args.output_dir / slug(str(row["species"])) / f"sample_{int(row['sample_index']):03d}.wav"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite")
        write_waveform(destination, waveform, config)
        species_counts[species] = species_counts.get(species, 0) + 1
        rows.append({
            "model": row.get("model", "generated"),
            "species": row["species"],
            "sample_index": int(row["sample_index"]),
            "relative_path": destination.relative_to(args.output_dir).as_posix(),
            "sample_rate": config.sample_rate,
            "num_samples": len(waveform),
            "peak_amplitude": float(np.max(np.abs(waveform))),
            "griffin_lim_iterations": args.iterations,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "audio_manifest.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Decoded {len(rows)} WAV files in {args.output_dir}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
