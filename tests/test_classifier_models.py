from __future__ import annotations

import torch

from bird_song.classifier.model import ARCHITECTURES, BirdSongCNN, build_classifier, count_trainable_parameters
from bird_song.config import DEFAULT_CLASSES, SpectrogramConfig
from bird_song.runtime import load_checkpoint


def test_every_architecture_produces_class_logits() -> None:
    inputs = torch.randn(2, 1, 128, 128)
    parameter_counts = []

    for architecture in ARCHITECTURES:
        model = build_classifier(architecture, num_classes=3)
        with torch.inference_mode():
            outputs = model(inputs)
        assert outputs.shape == (2, 3)
        assert model.metadata()["architecture"] == architecture
        parameter_counts.append(count_trainable_parameters(model))

    assert len(set(parameter_counts)) == len(ARCHITECTURES)


def test_new_architecture_checkpoint_round_trip(tmp_path) -> None:
    original = build_classifier("crnn", num_classes=len(DEFAULT_CLASSES), width=8, dropout=0.1)
    checkpoint_path = tmp_path / "crnn.pt"
    torch.save(
        {
            "format_version": 2,
            "model_state": original.state_dict(),
            "model_config": original.metadata(),
            "classes": list(DEFAULT_CLASSES),
            "spectrogram_config": SpectrogramConfig().to_dict(),
        },
        checkpoint_path,
    )

    loaded, classes, _, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    assert loaded.metadata() == original.metadata()
    assert classes == DEFAULT_CLASSES


def test_version_one_checkpoint_remains_loadable(tmp_path) -> None:
    original = BirdSongCNN(num_classes=len(DEFAULT_CLASSES), width=8, dropout=0.1)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": original.state_dict(),
            "model_config": {"num_classes": len(DEFAULT_CLASSES), "width": 8, "dropout": 0.1},
            "classes": list(DEFAULT_CLASSES),
            "spectrogram_config": SpectrogramConfig().to_dict(),
        },
        checkpoint_path,
    )

    loaded, classes, _, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    assert loaded.metadata()["architecture"] == "residual_cnn"
    assert classes == DEFAULT_CLASSES
