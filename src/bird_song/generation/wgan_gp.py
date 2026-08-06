from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WGANConfig:
    """Contract for a class-conditional WGAN-GP on 128 x 128 spectrograms."""

    image_size: int = 128
    channels: int = 1
    num_classes: int = 3
    latent_dim: int = 128
    base_channels: int = 64
    label_dim: int = 64
    critic_base_channels: int = 64
    gradient_penalty_weight: float = 10.0
    critic_steps: int = 5

    def __post_init__(self) -> None:
        if self.image_size != 128:
            raise ValueError("The shared generator branch currently requires image_size=128")
        if self.channels < 1 or self.num_classes < 1 or self.latent_dim < 1:
            raise ValueError("channels, num_classes, and latent_dim must be positive")
        if self.base_channels < 8 or self.critic_base_channels < 8:
            raise ValueError("base channel counts are too small")
        if self.label_dim < 1 or self.gradient_penalty_weight < 0 or self.critic_steps < 1:
            raise ValueError("label_dim, gradient penalty, and critic steps must be valid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "WGANConfig":
        return cls(**values)


class _GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalGenerator(nn.Module):
    """Class-conditional generator with local convolutional detail synthesis."""

    def __init__(self, config: WGANConfig) -> None:
        super().__init__()
        self.config = config
        c = config.base_channels
        self.label_embedding = nn.Embedding(config.num_classes, config.label_dim)
        self.input = nn.Sequential(
            nn.Linear(config.latent_dim + config.label_dim, c * 16 * 4 * 4),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _GeneratorBlock(c * 16, c * 8),
            _GeneratorBlock(c * 8, c * 4),
            _GeneratorBlock(c * 4, c * 2),
            _GeneratorBlock(c * 2, c),
            _GeneratorBlock(c, max(c // 2, 16)),
        )
        self.output = nn.Sequential(
            nn.Conv2d(max(c // 2, 16), c // 4, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 4, config.channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"Expected noise shaped (batch, {self.config.latent_dim}), got {tuple(noise.shape)}"
            )
        _validate_labels(labels, noise.shape[0], self.config.num_classes)
        condition = self.label_embedding(labels)
        x = self.input(torch.cat((noise, condition), dim=1))
        x = x.reshape(noise.shape[0], self.config.base_channels * 16, 4, 4)
        return self.output(self.blocks(x))

    @torch.inference_mode()
    def sample(
        self,
        labels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        _validate_labels(labels, labels.shape[0], self.config.num_classes)
        noise = torch.randn(
            labels.shape[0],
            self.config.latent_dim,
            device=labels.device,
            dtype=self.label_embedding.weight.dtype,
            generator=generator,
        )
        return self(noise, labels)


class _CriticBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(max(1, min(32, out_channels // 8)), out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, min(32, out_channels // 8)), out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalCritic(nn.Module):
    """Projection critic; it returns one scalar per spectrogram."""

    def __init__(self, config: WGANConfig) -> None:
        super().__init__()
        self.config = config
        c = config.critic_base_channels
        self.features = nn.Sequential(
            nn.Conv2d(config.channels, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _CriticBlock(c, c * 2),
            _CriticBlock(c * 2, c * 4),
            _CriticBlock(c * 4, c * 8),
            _CriticBlock(c * 8, c * 16),
        )
        self.output = nn.Linear(c * 16 * 4 * 4, 1)
        self.label_embedding = nn.Embedding(config.num_classes, c * 16 * 4 * 4)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        expected = (self.config.channels, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected images shaped (batch, {expected}), got {tuple(images.shape)}")
        _validate_labels(labels, images.shape[0], self.config.num_classes)
        features = self.features(images).flatten(1)
        score = self.output(features).squeeze(1)
        projection = (features * self.label_embedding(labels)).sum(dim=1)
        return score + projection / features.shape[1] ** 0.5


def _validate_labels(labels: torch.Tensor, batch_size: int, num_classes: int) -> None:
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError(f"Expected labels shaped ({batch_size},), got {tuple(labels.shape)}")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(f"Labels must be in [0, {num_classes - 1}]")


def gradient_penalty(
    critic: ConditionalCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the WGAN-GP penalty in float32, even under autocast."""
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device, dtype=torch.float32)
    interpolated = (alpha * real.float() + (1.0 - alpha) * fake.float()).requires_grad_(True)
    scores = critic(interpolated, labels).float()
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = gradients.flatten(1).norm(2, dim=1)
    return (norm - 1.0).square().mean()


def critic_loss(real_score: torch.Tensor, fake_score: torch.Tensor, penalty: torch.Tensor, weight: float) -> torch.Tensor:
    return fake_score.mean() - real_score.mean() + weight * penalty


def generator_loss(fake_score: torch.Tensor) -> torch.Tensor:
    return -fake_score.mean()


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
