from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bird_song import DEFAULT_CLASSES
from bird_song.generation.evaluation import detail_metrics, validate_generated
from bird_song.runtime import save_json
from bird_song.vocoder import VocoderMelScaler, VocoderSpectrogramConfig
from bird_song.vocoder_data import BigVGANMelDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_generated(manifest_path: Path, species: str) -> torch.Tensor:
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["species"].astype(str).eq(species)]
    paths = [manifest_path.parent / Path(str(row["scaled_mel_path"])) for _, row in rows.iterrows()]
    if len(paths) < 2:
        raise ValueError(f"Need at least two generated samples for {species} in {manifest_path}")
    arrays = [torch.from_numpy(np.load(path, allow_pickle=False)).float() for path in paths]
    return torch.stack(arrays).unsqueeze(1)


def diversity_metrics(images: torch.Tensor) -> dict[str, float]:
    flattened = images.float().flatten(1)
    distances = torch.pdist(flattened)
    return {
        "sample_count": float(images.shape[0]),
        "pairwise_l2_mean": float(distances.mean()),
        "pairwise_l2_median": float(distances.median()),
        "pairwise_l2_min": float(distances.min()),
        "sample_std": float(flattened.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated BigVGAN mels against validation detail statistics.")
    parser.add_argument("--generated-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/bigvgan_mels/mel_manifest.csv",
    )
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/bigvgan_mels")
    parser.add_argument(
        "--vocoder-config",
        type=Path,
        default=PROJECT_ROOT / "configs/bigvgan_spectrogram.json",
    )
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports/generator_comparison.json")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    vocoder_config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    scaler = VocoderMelScaler.from_json(args.cache_root / "scaler.json")
    real_dataset = BigVGANMelDataset(
        args.cache_manifest,
        args.cache_root,
        args.split,
        DEFAULT_CLASSES,
        vocoder_config,
        scaler,
    )
    real_by_species: dict[str, list[torch.Tensor]] = {name: [] for name in DEFAULT_CLASSES}
    for index in range(len(real_dataset)):
        mel, label, _ = real_dataset[index]
        real_by_species[DEFAULT_CLASSES[int(label)]].append(mel)
    real_tensors = {species: torch.stack(values) for species, values in real_by_species.items()}

    results: dict[str, object] = {
        "split": args.split,
        "representation": "bigvgan_normalized_logmel",
        "vocoder": vocoder_config.to_dict(),
        "scaler": scaler.to_dict(),
        "models": {},
    }
    for generated_manifest in args.generated_manifest:
        model_name = generated_manifest.parent.name
        model_result: dict[str, object] = {"manifest": str(generated_manifest), "species": {}}
        for species in DEFAULT_CLASSES:
            generated = load_generated(generated_manifest, species)
            expected = (1, vocoder_config.n_mels, vocoder_config.expected_frames)
            validity = validate_generated(generated, expected_shape=expected)
            model_result["species"][species] = {
                "validity": validity,
                "detail": detail_metrics(real_tensors[species], generated),
                "diversity": diversity_metrics(generated),
            }
        results["models"][model_name] = model_result

    save_json(args.output_json, results)
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation to {args.output_json}")


if __name__ == "__main__":
    main()
