from __future__ import annotations

import torch
from torch import nn


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

    def metadata(self) -> dict[str, float | int]:
        return {"num_classes": self.num_classes, "dropout": self.dropout, "width": self.width}
