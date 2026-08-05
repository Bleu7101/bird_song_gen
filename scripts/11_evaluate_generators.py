from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bird_song.config import SpectrogramConfig
from bird_song.generation.evaluation import detail_metrics, diversity_metrics, validate_generated
from bird_song.runtime import save_json
from bird_song.transformer.data import CachedSpectrogramDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated spectrogram detail and diversity without retraining the classifier.")
    parser.add_argument("--generated-manifest", type=Path, action="append", required=True, help="Repeat for each model manifest.")
    parser.add_argument("--cache-manifest", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms/spectrogram_manifest.csv")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "outputs/generator_evaluation.json")
    return parser.parse_args()


def load_generated(manifest_path: Path, species: str) -> torch.Tensor:
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["species"] == species]
    arrays = [torch.from_numpy(np.load(manifest_path.parent / Path(row["relative_path"]), allow_pickle=False)).float() for _, row in rows.iterrows()]
    if not arrays:
        raise ValueError(f"No generated samples for {species} in {manifest_path}")
    return torch.stack(arrays).unsqueeze(1)


def main() -> None:
    args = parse_args()
    config = SpectrogramConfig.from_json(args.spectrogram_config)
    if (config.n_mels, config.spectrogram_width) != (128, 128):
        raise ValueError("Evaluation currently targets the existing 128 x 128 branch")
    manifest = pd.read_csv(args.cache_manifest)
    classes = tuple(sorted(manifest["name"].unique()))
    dataset = CachedSpectrogramDataset(args.cache_manifest, args.cache_root, args.split, classes, 128, specaugment=False)
    real_by_species: dict[str, list[torch.Tensor]] = {name: [] for name in classes}
    for index in range(len(dataset)):
        image, label, _ = dataset[index]
        real_by_species[classes[label]].append(image)
    real_tensors = {species: torch.stack(images) for species, images in real_by_species.items()}
    results: dict[str, object] = {"split": args.split, "models": {}}
    for generated_manifest in args.generated_manifest:
        model_name = generated_manifest.parent.name
        model_result: dict[str, object] = {"manifest": str(generated_manifest), "species": {}}
        for species in classes:
            generated = load_generated(generated_manifest, species)
            real = real_tensors[species]
            model_result["species"][species] = {
                "validity": validate_generated(generated),
                "detail": detail_metrics(real, generated),
                "diversity": diversity_metrics(generated) if generated.shape[0] > 1 else {"sample_count": 1},
            }
        results["models"][model_name] = model_result
    save_json(args.output_json, results)
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation to {args.output_json}")


if __name__ == "__main__":
    main()
