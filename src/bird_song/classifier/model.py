from __future__ import annotations

import torch
from torch import nn


ARCHITECTURES = ("residual_cnn", "plain_cnn", "depthwise_cnn", "crnn")


def _validate_model_arguments(num_classes: int, dropout: float, width: int) -> None:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if width < 4:
        raise ValueError("width must be at least 4")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class BirdSongCNN(nn.Module):
    """Compact residual CNN for single-channel 128 x 128 log-mel inputs."""

    def __init__(self, num_classes: int, dropout: float = 0.30, width: int = 32) -> None:
        super().__init__()
        _validate_model_arguments(num_classes, dropout, width)
        self.num_classes = num_classes
        self.dropout = dropout
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, stride=2),
            ResidualBlock(width * 2, width * 2),
            ResidualBlock(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4, width * 4),
            ResidualBlock(width * 4, width * 8, stride=2),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 16, width * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(inputs))
        pooled = torch.cat((self.average_pool(features), self.maximum_pool(features)), dim=1)
        return self.classifier(pooled)

    def metadata(self) -> dict[str, str | float | int]:
        return {
            "architecture": "residual_cnn",
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "width": self.width,
        }


class ConvBlock(nn.Module):
    """Two convolutions followed by spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int | tuple[int, int] = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class PlainBirdSongCNN(nn.Module):
    """Conventional VGG-style CNN without residual connections."""

    def __init__(self, num_classes: int, dropout: float = 0.30, width: int = 32) -> None:
        super().__init__()
        _validate_model_arguments(num_classes, dropout, width)
        self.num_classes = num_classes
        self.dropout = dropout
        self.width = width
        self.features = nn.Sequential(
            ConvBlock(1, width),
            ConvBlock(width, width * 2),
            ConvBlock(width * 2, width * 4),
            ConvBlock(width * 4, width * 8),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 16, width * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = torch.cat((self.average_pool(features), self.maximum_pool(features)), dim=1)
        return self.classifier(pooled)

    def metadata(self) -> dict[str, str | float | int]:
        return {
            "architecture": "plain_cnn",
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "width": self.width,
        }


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DepthwiseBirdSongCNN(nn.Module):
    """MobileNet-style CNN testing a substantially lower-capacity model."""

    def __init__(self, num_classes: int, dropout: float = 0.30, width: int = 32) -> None:
        super().__init__()
        _validate_model_arguments(num_classes, dropout, width)
        self.num_classes = num_classes
        self.dropout = dropout
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(width, width),
            DepthwiseSeparableBlock(width, width * 2, stride=2),
            DepthwiseSeparableBlock(width * 2, width * 2),
            DepthwiseSeparableBlock(width * 2, width * 4, stride=2),
            DepthwiseSeparableBlock(width * 4, width * 4),
            DepthwiseSeparableBlock(width * 4, width * 8, stride=2),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 16, width * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(inputs))
        pooled = torch.cat((self.average_pool(features), self.maximum_pool(features)), dim=1)
        return self.classifier(pooled)

    def metadata(self) -> dict[str, str | float | int]:
        return {
            "architecture": "depthwise_cnn",
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "width": self.width,
        }


class BirdSongCRNN(nn.Module):
    """CNN + bidirectional GRU that explicitly models time order."""

    def __init__(self, num_classes: int, dropout: float = 0.30, width: int = 32) -> None:
        super().__init__()
        _validate_model_arguments(num_classes, dropout, width)
        self.num_classes = num_classes
        self.dropout = dropout
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )
        # The last block downsamples frequency only so the GRU retains 16 time steps.
        self.features = nn.Sequential(
            ConvBlock(width, width, pool_size=2),
            ConvBlock(width, width * 2, pool_size=2),
            ConvBlock(width * 2, width * 4, pool_size=(2, 1)),
        )
        self.recurrent = nn.GRU(
            input_size=width * 4,
            hidden_size=width * 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width * 8, width * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(inputs)).mean(dim=2).transpose(1, 2)
        sequence, _ = self.recurrent(features)
        pooled = torch.cat((sequence.mean(dim=1), sequence.amax(dim=1)), dim=1)
        return self.classifier(pooled)

    def metadata(self) -> dict[str, str | float | int]:
        return {
            "architecture": "crnn",
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "width": self.width,
        }


def build_classifier(
    architecture: str,
    num_classes: int,
    dropout: float = 0.30,
    width: int = 32,
) -> nn.Module:
    """Build a classifier from the canonical architecture name stored in checkpoints."""
    models: dict[str, type[nn.Module]] = {
        "residual_cnn": BirdSongCNN,
        "plain_cnn": PlainBirdSongCNN,
        "depthwise_cnn": DepthwiseBirdSongCNN,
        "crnn": BirdSongCRNN,
    }
    try:
        model_class = models[architecture]
    except KeyError as error:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}") from error
    return model_class(num_classes=num_classes, dropout=dropout, width=width)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
