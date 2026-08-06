from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from bird_song.audio import load_generated_spectrogram
from bird_song.config import SpectrogramConfig
from bird_song.data import ManifestDataset
from bird_song.classifier.model import BirdSongCNN
from bird_song.runtime import load_checkpoint


def test_audio_to_model(tmp_path: Path) -> None:
    config = SpectrogramConfig()
    dataset_root = tmp_path / "dataset"
    wav_dir = dataset_root / "wavfiles"
    wav_dir.mkdir(parents=True)
    (dataset_root / "bird_songs_metadata.csv").write_text("id,name,filename\n", encoding="utf-8")
    time = np.arange(config.sample_rate * 2, dtype=np.float32) / config.sample_rate
    sf.write(wav_dir / "tone.wav", np.sin(2 * np.pi * 1_000 * time), config.sample_rate)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"name": "Test Bird", "relative_wav_path": "wavfiles/tone.wav"}]).to_csv(manifest, index=False)

    spec, label, _ = ManifestDataset(manifest, dataset_root, ("Test Bird",), config)[0]
    assert spec.shape == (1, 128, 128)
    assert label == 0
    assert torch.isfinite(spec).all()
    assert BirdSongCNN(2)(spec.unsqueeze(0)).shape == (1, 2)


def test_generated_spectrogram_ranges(tmp_path: Path) -> None:
    path = tmp_path / "generated.npy"
    np.save(path, np.linspace(-80, 0, 64 * 96, dtype=np.float32).reshape(64, 96))
    spec = load_generated_spectrogram(path, SpectrogramConfig())
    assert spec.shape == (1, 128, 128)
    assert float(spec.min()) >= -1.0
    assert float(spec.max()) <= 1.0


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = SpectrogramConfig()
    model = BirdSongCNN(3)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.metadata(),
            "classes": ["A", "B", "C"],
            "spectrogram_config": config.to_dict(),
        },
        path,
    )
    loaded, classes, loaded_config, _ = load_checkpoint(path, torch.device("cpu"))
    assert classes == ("A", "B", "C")
    assert loaded_config == config
    assert loaded(torch.zeros(2, 1, 128, 128)).shape == (2, 3)


def test_manifest_rejects_empty_file(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "wavfiles").mkdir(parents=True)
    manifest = tmp_path / "empty.csv"
    pd.DataFrame(columns=["name", "relative_wav_path"]).to_csv(manifest, index=False)
    try:
        ManifestDataset(manifest, dataset_root, ("Test Bird",), SpectrogramConfig())
    except ValueError as error:
        assert "empty" in str(error).lower()
    else:
        raise AssertionError("Empty manifest was accepted")
