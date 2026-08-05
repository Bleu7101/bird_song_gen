from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from bird_song import DEFAULT_CLASSES
from bird_song.generation.evaluation import validate_generated
from bird_song.generation.wgan_gp import ConditionalGenerator, WGANConfig
from bird_song.metrics import waveform_diagnostics
from bird_song.runtime import choose_device, save_json, seed_everything
from bird_song.vocoder import VocoderMelScaler, VocoderSpectrogramConfig, load_bigvgan, vocoder_mel_to_waveform


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WGAN BigVGAN mels and decode them to audio.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs/wgan_gp_bigvgan/best_generator.pt")
    parser.add_argument("--bigvgan-source", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/wgan_gp_bigvgan_audio")
    parser.add_argument("--samples-per-species", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples_per_species < 1:
        raise ValueError("samples-per-species must be positive")
    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = WGANConfig.from_dict(checkpoint["model_config"])
    vocoder_config = VocoderSpectrogramConfig.from_dict(checkpoint["vocoder_config"])
    scaler = VocoderMelScaler.from_dict(checkpoint["scaler"])
    classes = tuple(checkpoint["classes"])
    if classes != DEFAULT_CLASSES:
        raise ValueError(f"Unexpected class order: {classes}")
    generator = ConditionalGenerator(model_config).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()
    source = args.bigvgan_source or PROJECT_ROOT / "external" / vocoder_config.model_id.rsplit("/", 1)[-1]
    vocoder = load_bigvgan(source, device, vocoder_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "generated_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    scaled_for_diversity: list[torch.Tensor] = []
    scaled_by_species: dict[str, list[torch.Tensor]] = {name: [] for name in classes}
    for class_index, class_name in enumerate(classes):
        labels = torch.full((args.samples_per_species,), class_index, dtype=torch.long, device=device)
        scaled_samples = generator.sample(labels, generator=random_generator)
        validate_generated(scaled_samples)
        # The WGAN keeps an explicit channel dimension; the public mel/vocoder
        # contract is a single [80, 256] array per sample.
        scaled_samples_cpu = scaled_samples.detach().cpu()[:, 0]
        raw_samples = scaler.denormalize(scaled_samples_cpu)
        for sample_index, (scaled, raw) in enumerate(zip(scaled_samples_cpu, raw_samples)):
            scaled_for_diversity.append(scaled.flatten())
            scaled_by_species[class_name].append(scaled.flatten())
            species_dir = args.output_dir / slug(class_name)
            mel_path = species_dir / f"sample_{sample_index:03d}.npy"
            scaled_path = species_dir / f"sample_{sample_index:03d}_scaled.npy"
            wav_path = species_dir / f"sample_{sample_index:03d}.wav"
            if not args.overwrite and any(path.exists() for path in (mel_path, scaled_path, wav_path)):
                raise FileExistsError(f"Refusing to overwrite generated sample {mel_path}")
            species_dir.mkdir(parents=True, exist_ok=True)
            np.save(mel_path, raw.numpy().astype(np.float32), allow_pickle=False)
            np.save(scaled_path, scaled.numpy().astype(np.float32), allow_pickle=False)
            waveform = vocoder_mel_to_waveform(raw, vocoder, vocoder_config)
            sf.write(wav_path, waveform.numpy(), vocoder_config.sample_rate, subtype="PCM_16")
            diagnostic = waveform_diagnostics(waveform, vocoder_config.num_samples)
            rows.append(
                {
                    "model": "conditional_wgan_gp_bigvgan_mel_generator",
                    "species": class_name,
                    "label_id": class_index,
                    "sample_index": sample_index,
                    "raw_mel_path": mel_path.relative_to(args.output_dir).as_posix(),
                    "scaled_mel_path": scaled_path.relative_to(args.output_dir).as_posix(),
                    "wav_path": wav_path.relative_to(args.output_dir).as_posix(),
                    "raw_mel_min": float(raw.min()),
                    "raw_mel_max": float(raw.max()),
                    "scaled_saturation_fraction": float((scaled.abs() >= 0.999).float().mean()),
                    **diagnostic,
                }
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = bool(all(int(row["valid"]) for row in rows))
    stacked = torch.stack(scaled_for_diversity)
    pairwise_distances = torch.pdist(stacked)
    diversity_by_species = {
        species: float(torch.pdist(torch.stack(values)).mean()) if len(values) > 1 else 0.0
        for species, values in scaled_by_species.items()
    }
    summary = {
        "status": "integration_pass" if valid else "integration_fail",
        "generated_samples": len(rows),
        "samples_per_species": args.samples_per_species,
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_config": model_config.to_dict(),
        "vocoder_config": vocoder_config.to_dict(),
        "scaler": scaler.to_dict(),
        "all_waveforms_valid": valid,
        "maximum_clipped_fraction": max(float(row["clipped_fraction"]) for row in rows),
        "silent_samples": sum(int(row["silent"]) for row in rows),
        "mel_diagnostics": {
            "minimum": min(float(row["raw_mel_min"]) for row in rows),
            "maximum": max(float(row["raw_mel_max"]) for row in rows),
            "mean_scaled_saturation_fraction": sum(float(row["scaled_saturation_fraction"]) for row in rows) / len(rows),
            "mean_pairwise_scaled_l2_distance": float(pairwise_distances.mean()),
            "mean_pairwise_scaled_l2_distance_by_species": diversity_by_species,
        },
        "note": "Output validity and conditioning labels do not prove acoustic realism; listen to the local WAVs.",
    }
    save_json(args.output_dir / "generation_summary.json", summary)
    print(f"Generation status: {summary['status']}")
    print(f"Generated {len(rows)} decoded samples in {args.output_dir}")


if __name__ == "__main__":
    main()
