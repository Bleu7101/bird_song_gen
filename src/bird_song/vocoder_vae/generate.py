from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from bird_song.runtime import choose_device, seed_everything
from bird_song.vocoder import (
    VocoderMelNormalizer,
    VocoderSpectrogramConfig,
    load_bigvgan,
    vocoder_mel_to_waveform,
)
from bird_song.vocoder_data import VocoderSpectrogramDataset, make_vocoder_loader
from bird_song.vocoder_vae.model import ConditionalVocoderVAE, VocoderVAEConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BigVGAN-compatible VAE log-mels and waveforms.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs/vocoder_vae/best.pt")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/vocoder_spectrograms/vocoder_spectrogram_manifest.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/vocoder_spectrograms",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/vocoder_vae")
    parser.add_argument("--samples-per-species", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bigvgan-source", type=Path, default=None)
    parser.add_argument("--bigvgan-model", default=None)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def fit_class_conditional_prior(
    model: ConditionalVocoderVAE,
    loader,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    shape = (num_classes, *model.config.latent_shape)
    sums = torch.zeros(shape, device=device)
    second_moments = torch.zeros_like(sums)
    counts = torch.zeros(num_classes, device=device)
    for spectrograms, labels, _ in loader:
        spectrograms = spectrograms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mu, logvar = model.encode(spectrograms, labels)
        sums.index_add_(0, labels, mu)
        second_moments.index_add_(0, labels, mu.square() + logvar.exp())
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    if bool((counts == 0).any()):
        raise RuntimeError("Every class needs at least one training example to fit the prior")
    means = sums / counts[:, None, None, None]
    variances = second_moments / counts[:, None, None, None] - means.square()
    return means, variances.clamp_min(1e-4).sqrt().clamp_max(3.0)


def main() -> None:
    args = parse_args()
    if args.samples_per_species < 1 or args.batch_size < 1:
        raise ValueError("samples-per-species and batch-size must be positive")
    if args.temperature < 0 or args.workers < 0:
        raise ValueError("temperature and workers cannot be negative")
    manifest_path = args.output_dir / "generated_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite")

    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = VocoderVAEConfig.from_dict(checkpoint["model_config"])
    vocoder_config = VocoderSpectrogramConfig.from_dict(checkpoint["vocoder_config"])
    normalizer = VocoderMelNormalizer.from_dict(checkpoint["normalizer"])
    classes = tuple(checkpoint["classes"])
    model = ConditionalVocoderVAE(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    train_set = VocoderSpectrogramDataset(
        args.cache_manifest,
        args.cache_root,
        "train",
        classes,
        vocoder_config,
        normalizer,
    )
    train_loader = make_vocoder_loader(train_set, args.batch_size, args.workers)
    prior_mean, prior_std = fit_class_conditional_prior(model, train_loader, len(classes), device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": prior_mean.cpu(),
            "std": prior_std.cpu(),
            "temperature": args.temperature,
            "classes": list(classes),
            "fitted_split": "train",
        },
        args.output_dir / "class_conditional_aggregated_prior.pt",
    )

    vocoder = None
    if not args.skip_audio:
        default_source = PROJECT_ROOT / "external" / vocoder_config.model_id.rsplit("/", 1)[-1]
        source = args.bigvgan_source or default_source
        model_reference: str | Path = args.bigvgan_model or (
            source if (source / "bigvgan_generator.pt").is_file() else vocoder_config.model_id
        )
        vocoder = load_bigvgan(model_reference, device, vocoder_config, source_dir=source)

    random_generator = torch.Generator(device=device.type).manual_seed(args.seed + 100)
    rows: list[dict] = []
    preview: list[tuple[str, torch.Tensor]] = []
    for label_id, class_name in enumerate(classes):
        species_slug = slug(class_name)
        normalized_dir = args.output_dir / "normalized_mel" / species_slug
        raw_dir = args.output_dir / "raw_vocoder_mel" / species_slug
        audio_dir = args.output_dir / "generated_audio" / species_slug
        normalized_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if vocoder is not None:
            audio_dir.mkdir(parents=True, exist_ok=True)
        species_samples = []
        for start in range(0, args.samples_per_species, args.batch_size):
            count = min(args.batch_size, args.samples_per_species - start)
            labels = torch.full((count,), label_id, dtype=torch.long, device=device)
            epsilon = torch.randn(
                (count, *model_config.latent_shape),
                generator=random_generator,
                device=device,
            )
            latent = prior_mean[labels] + args.temperature * prior_std[labels] * epsilon
            normalized_batch = model.decode(latent, labels).cpu()
            raw_batch = normalizer.denormalize(normalized_batch)
            waveforms = None
            if vocoder is not None:
                waveforms = vocoder_mel_to_waveform(raw_batch[:, 0], vocoder, vocoder_config)
                if waveforms.ndim == 1:
                    waveforms = waveforms.unsqueeze(0)
            for offset in range(count):
                sample_index = start + offset
                filename = f"sample_{sample_index:03d}"
                normalized_path = normalized_dir / f"{filename}.npy"
                raw_path = raw_dir / f"{filename}.npy"
                audio_path = audio_dir / f"{filename}.wav" if waveforms is not None else None
                for destination in (normalized_path, raw_path, audio_path):
                    if destination is not None and destination.exists() and not args.overwrite:
                        raise FileExistsError(f"Refusing to overwrite {destination}")
                np.save(
                    normalized_path,
                    normalized_batch[offset, 0].numpy().astype(np.float32),
                    allow_pickle=False,
                )
                np.save(
                    raw_path,
                    raw_batch[offset, 0].numpy().astype(np.float32),
                    allow_pickle=False,
                )
                if audio_path is not None:
                    sf.write(
                        audio_path,
                        waveforms[offset].numpy().astype(np.float32),
                        vocoder_config.sample_rate,
                    )
                rows.append(
                    {
                        "model": "spatial_conditional_vocoder_vae",
                        "species": class_name,
                        "label_id": label_id,
                        "sample_index": sample_index,
                        "normalized_mel_path": normalized_path.relative_to(args.output_dir).as_posix(),
                        "raw_vocoder_mel_path": raw_path.relative_to(args.output_dir).as_posix(),
                        "audio_path": (
                            audio_path.relative_to(args.output_dir).as_posix() if audio_path is not None else ""
                        ),
                        "temperature": args.temperature,
                        "seed": args.seed,
                    }
                )
                if sample_index < 4:
                    species_samples.append(raw_batch[offset, 0].clone())
        preview.append((class_name, torch.stack(species_samples)))

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if not args.no_figure:
        figure, axes = plt.subplots(len(classes), 4, figsize=(13, 3 * len(classes)), squeeze=False)
        for row_index, (class_name, samples) in enumerate(preview):
            for column_index, sample in enumerate(samples):
                axes[row_index, column_index].imshow(sample, origin="lower", aspect="auto", cmap="magma")
                axes[row_index, column_index].set_xticks([])
                axes[row_index, column_index].set_yticks([])
                if column_index == 0:
                    axes[row_index, column_index].set_ylabel(class_name)
        figure.suptitle(f"Vocoder VAE raw log-mel samples (temperature={args.temperature:g})")
        figure.tight_layout()
        figure.savefig(args.output_dir / "conditional_samples.png", dpi=160)
        plt.close(figure)
    print(f"Generated {len(rows)} samples in {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
