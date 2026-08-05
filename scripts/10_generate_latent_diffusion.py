from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bird_song.generation.diffusion import ConditionalLatentDenoiser, LatentDiffusionConfig, sample_latents
from bird_song.generation.evaluation import validate_generated
from bird_song.generation.vqgan import ConditionalVQGAN, VQGANConfig
from bird_song.runtime import choose_device, save_json, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spectrograms with latent diffusion plus a VQGAN decoder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/latent_diffusion")
    parser.add_argument("--samples-per-species", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_species < 1:
        raise ValueError("samples-per-species must be positive")
    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vq_checkpoint = torch.load(args.vqgan_checkpoint, map_location=device, weights_only=True)
    diffusion_config = LatentDiffusionConfig.from_dict(checkpoint["model_config"])
    vq_config = VQGANConfig.from_dict(vq_checkpoint["model_config"])
    model = ConditionalLatentDenoiser(diffusion_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    vqgan = ConditionalVQGAN(vq_config).to(device)
    vqgan.load_state_dict(vq_checkpoint["model_state"])
    vqgan.eval()
    classes = tuple(checkpoint["classes"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "generated_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite")
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(len(classes), min(args.samples_per_species, 8), figsize=(18, 2.8 * len(classes)), squeeze=False, constrained_layout=True)
    for class_index, class_name in enumerate(classes):
        labels = torch.full((args.samples_per_species,), class_index, dtype=torch.long, device=device)
        latents = sample_latents(model, labels, generator=random_generator)
        with torch.no_grad():
            samples = vqgan.decode_latents(latents, labels).cpu()
        save_json(args.output_dir / f"{slug(class_name)}_summary.json", validate_generated(samples))
        species_dir = args.output_dir / slug(class_name)
        species_dir.mkdir(parents=True, exist_ok=True)
        for sample_index, sample in enumerate(samples):
            path = species_dir / f"sample_{sample_index:03d}.npy"
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {path}")
            np.save(path, sample.squeeze(0).numpy().astype(np.float32), allow_pickle=False)
            rows.append({"model": "conditional_vqgan_latent_diffusion", "species": class_name, "label_id": class_index, "sample_index": sample_index, "relative_path": path.relative_to(args.output_dir).as_posix(), "seed": args.seed})
            if sample_index < axes.shape[1] and not args.no_figure:
                axes[class_index, sample_index].imshow(sample.squeeze(0), origin="lower", aspect="auto", cmap="magma", vmin=-1, vmax=1)
                axes[class_index, sample_index].set_xticks([])
                axes[class_index, sample_index].set_yticks([])
        if not args.no_figure:
            axes[class_index, 0].set_ylabel(class_name)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_figure:
        figure.suptitle("VQGAN latent diffusion samples")
        figure.savefig(args.output_dir / "conditional_samples.png", dpi=160)
        plt.close(figure)
    print(f"Generated {len(rows)} spectrograms in {args.output_dir}")


if __name__ == "__main__":
    main()
