from __future__ import annotations

import argparse
import platform
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio

from bird_song.config import SpectrogramConfig
from bird_song.data import ManifestDataset, resolve_dataset_root
from bird_song.classifier.model import BirdSongCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dependencies, CUDA, manifests, audio, and one model forward pass.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchAudio: {torchaudio.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    manifests = {name: pd.read_csv(PROJECT_ROOT / f"manifests/full_dataset_{name}.csv") for name in ("train", "validation", "test")}
    id_sets = {name: set(frame["id"].astype(str)) for name, frame in manifests.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            raise ValueError(f"Recording-ID leakage between {left} and {right}: {len(overlap)} IDs")
    missing = [root / relative for frame in manifests.values() for relative in frame["relative_wav_path"] if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest audio files are missing; first: {missing[0]}")

    classes = tuple(sorted(manifests["train"]["name"].unique()))
    config = SpectrogramConfig.from_json(PROJECT_ROOT / "configs/spectrogram.json")
    dataset = ManifestDataset(PROJECT_ROOT / "manifests/full_dataset_validation.csv", root, classes, config)
    spec, label, path = dataset[0]
    info = sf.info(path)
    output = BirdSongCNN(len(classes))(spec.unsqueeze(0))
    print(f"Dataset: {root} ({sum(len(frame) for frame in manifests.values())} selected clips)")
    print(f"Sample audio: {info.samplerate} Hz, {info.duration:.2f} s, {info.channels} channel(s)")
    print(f"Spectrogram: {tuple(spec.shape)}; label={classes[label]}")
    print(f"Model output: {tuple(output.shape)}")
    print("Setup validation passed. This command did not train the model.")


if __name__ == "__main__":
    main()
