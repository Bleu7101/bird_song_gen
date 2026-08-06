from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bird_song.runtime import choose_device, seed_everything
from bird_song.transformer.model import ConditionalSpectrogramTransformer, TransformerGeneratorConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate species-conditioned normalized log-mel images with a trained transformer."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/autoregressive_transformer")
    parser.add_argument("--samples-per-species", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_species < 1:
        raise ValueError("samples-per-species must be positive")
    if args.temperature < 0:
        raise ValueError("temperature cannot be negative")
    seed_everything(args.seed)
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = TransformerGeneratorConfig.from_dict(checkpoint["model_config"])
    classes = tuple(checkpoint["classes"])
    model = ConditionalSpectrogramTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    manifest_path = args.output_dir / "generated_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    generated_by_species: list[tuple[str, torch.Tensor]] = []
    rows = []

    for class_index, class_name in enumerate(classes):
        labels = torch.full(
            (args.samples_per_species,),
            class_index,
            dtype=torch.long,
            device=device,
        )
        samples = model.generate(labels, temperature=args.temperature, generator=random_generator).cpu()
        generated_by_species.append((class_name, samples))
        species_dir = args.output_dir / slug(class_name)
        species_dir.mkdir(parents=True, exist_ok=True)
        for sample_index, sample in enumerate(samples):
            destination = species_dir / f"sample_{sample_index:03d}.npy"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {destination}")
            np.save(destination, sample.squeeze(0).numpy().astype(np.float32), allow_pickle=False)
            rows.append(
                {
                    "model": "conditional_autoregressive_spectrogram_transformer",
                    "species": class_name,
                    "label_id": class_index,
                    "sample_index": sample_index,
                    "relative_path": destination.relative_to(args.output_dir).as_posix(),
                    "temperature": args.temperature,
                    "seed": args.seed,
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if not args.no_figure:
        figure, axes = plt.subplots(
            len(classes),
            args.samples_per_species,
            figsize=(2.4 * args.samples_per_species, 2.6 * len(classes)),
            squeeze=False,
            constrained_layout=True,
        )
        for row_index, (class_name, samples) in enumerate(generated_by_species):
            for column_index, sample in enumerate(samples):
                axis = axes[row_index, column_index]
                axis.imshow(sample.squeeze(0), origin="lower", aspect="auto", cmap="magma", vmin=-1, vmax=1)
                axis.set_xticks([])
                axis.set_yticks([])
                if column_index == 0:
                    axis.set_ylabel(class_name)
        figure.suptitle(
            f"Autoregressive transformer samples (temperature={args.temperature:g})",
            fontsize=14,
        )
        figure.savefig(args.output_dir / "conditional_samples.png", dpi=160)
        plt.close(figure)

    print(f"Generated {len(rows)} spectrograms in {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
