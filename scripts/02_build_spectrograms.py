from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bird_song.audio import LogMelTransform, load_waveform
from bird_song.config import SpectrogramConfig
from bird_song.data import resolve_dataset_root
from bird_song.spectrogram_cache import array_sha256, cache_object_path, canonicalize_spectrogram_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the shared normalized log-mel cache for Stages 3-5.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a smoke test.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    rows = pd.read_csv(args.manifest)
    required = {"split", "name", "filename", "relative_wav_path"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    if rows.empty:
        raise ValueError("No manifest rows selected")
    cache_manifest = args.output_dir / "spectrogram_manifest.csv"
    if cache_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{cache_manifest} already exists; pass --overwrite to rebuild this cache")

    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    config = SpectrogramConfig.from_json(args.spectrogram_config)
    transform = LogMelTransform(config, training=False)
    output_rows = []
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="Spectrograms"):
        waveform = load_waveform(dataset_root / row.relative_wav_path, config, training=False)
        spectrogram = transform(waveform).squeeze(0).numpy().astype(np.float32, copy=False)
        digest = array_sha256(spectrogram)
        relative_output = cache_object_path(digest)
        destination = args.output_dir / relative_output
        if args.overwrite or not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, spectrogram, allow_pickle=False)
        output = row._asdict()
        output["relative_spectrogram_path"] = relative_output.as_posix()
        output["spectrogram_sha256"] = digest
        output_rows.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(cache_manifest, index=False)
    canonicalize_spectrogram_cache(args.output_dir, config, args.spectrogram_config.resolve(), apply=True)
    print(f"Saved {len(output_rows)} spectrograms with shape ({config.n_mels}, {config.spectrogram_width})")
    print(f"Cache manifest: {cache_manifest}")


if __name__ == "__main__":
    main()
