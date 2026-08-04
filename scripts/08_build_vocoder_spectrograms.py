from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bird_song.data import resolve_dataset_root
from bird_song.runtime import choose_device, save_json
from bird_song.vocoder import (
    VocoderMelNormalizer,
    VocoderSpectrogramConfig,
    load_vocoder_waveform,
    waveform_to_vocoder_mel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the isolated BigVGAN-compatible raw log-mel cache.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--vocoder-config",
        type=Path,
        default=PROJECT_ROOT / "configs/vocoder_spectrogram.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/vocoder_spectrograms",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    rows = pd.read_csv(args.manifest)
    required = {"split", "name", "filename", "relative_wav_path"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    if rows.empty:
        raise ValueError("No manifest rows selected")
    if "train" not in set(rows["split"]):
        raise ValueError("Selected rows contain no training examples; training-only normalization is impossible")

    manifest_path = args.output_dir / "vocoder_spectrogram_manifest.csv"
    stats_path = args.output_dir / "normalization_stats.json"
    existing_contracts = [path for path in (manifest_path, stats_path) if path.exists()]
    if existing_contracts and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {[path.name for path in existing_contracts]}; pass --overwrite"
        )

    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    device = choose_device(args.device)
    config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    output_rows: list[dict] = []
    train_sum = 0.0
    train_sum_squares = 0.0
    train_count = 0

    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="Vocoder log-mels"):
        species_slug = getattr(row, "species_slug", None) or slug(row.name)
        relative_output = Path(row.split) / species_slug / f"{Path(row.filename).stem}.npy"
        destination = args.output_dir / relative_output
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Raw log-mel already exists: {destination}")
        waveform = load_vocoder_waveform(dataset_root / row.relative_wav_path, config)
        raw_logmel = (
            waveform_to_vocoder_mel(waveform.to(device, non_blocking=True), config)[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, raw_logmel, allow_pickle=False)
        if row.split == "train":
            values = raw_logmel.astype(np.float64, copy=False)
            train_sum += float(values.sum())
            train_sum_squares += float(np.square(values).sum())
            train_count += values.size
        output = row._asdict()
        output["relative_vocoder_mel_path"] = relative_output.as_posix()
        output_rows.append(output)

    mean = train_sum / train_count
    variance = max(train_sum_squares / train_count - mean * mean, 1e-12)
    normalizer = VocoderMelNormalizer(mean=mean, std=float(np.sqrt(variance)), count=train_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(manifest_path, index=False)
    save_json(stats_path, normalizer.to_dict())
    save_json(args.output_dir / "vocoder_config.json", config.to_dict())
    print(f"Saved {len(output_rows)} raw log-mels shaped ({config.n_mels}, {config.expected_frames})")
    print(f"Mel frontend device: {device}")
    print(f"Training normalizer: mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    print(f"Cache manifest: {manifest_path}")


if __name__ == "__main__":
    main()
