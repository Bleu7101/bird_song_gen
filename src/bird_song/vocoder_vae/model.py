from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VocoderVAEConfig:
    """Spatial conditional VAE contract for 80 x 256 BigVGAN log-mels."""

    image_height: int = 80
    image_width: int = 256
    input_channels: int = 1
    latent_channels: int = 16
    class_embed_dim: int = 64
    base_channels: int = 32
    num_classes: int = 3
    downsample_stages: int = 4

    def __post_init__(self) -> None:
        values = asdict(self)
        invalid = [name for name, value in values.items() if value < 1]
        if invalid:
            raise ValueError(f"VAE config values must be positive: {invalid}")
        factor = self.downsample_factor
        if self.image_height % factor or self.image_width % factor:
            raise ValueError(
                f"Image shape {(self.image_height, self.image_width)} must be divisible by {factor}"
            )

    @property
    def downsample_factor(self) -> int:
        return 2**self.downsample_stages

    @property
    def latent_height(self) -> int:
        return self.image_height // self.downsample_factor

    @property
    def latent_width(self) -> int:
        return self.image_width // self.downsample_factor

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self.latent_channels, self.latent_height, self.latent_width

    @property
    def latent_dim(self) -> int:
        return self.latent_channels * self.latent_height * self.latent_width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VocoderVAEConfig":
        return cls(**values)

    @classmethod
    def from_json(cls, path: Path) -> "VocoderVAEConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.film = nn.Linear(condition_dim, out_channels * 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.norm2(hidden)
        scale, shift = self.film(condition).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale[:, :, None, None]))
        hidden = hidden + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return residual + hidden


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(inputs, scale_factor=2.0, mode="bilinear", align_corners=False))


class ConditionalVocoderVAE(nn.Module):
    """Detail-aware class-conditional VAE with a 16 x 5 x 16 spatial latent."""

    def __init__(self, config: VocoderVAEConfig) -> None:
        super().__init__()
        if config.downsample_stages != 4:
            raise ValueError("The published spatial VAE v2 architecture has exactly four downsample stages")
        self.config = config
        self.class_embedding = nn.Embedding(config.num_classes, config.class_embed_dim)
        b = config.base_channels
        channels = [b, b * 2, b * 4, b * 6, b * 8]

        self.encoder_stem = nn.Conv2d(config.input_channels, b, kernel_size=3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            [ConditionalResBlock(channel, channel, config.class_embed_dim) for channel in channels]
        )
        self.downsamples = nn.ModuleList(
            [Downsample(channels[index], channels[index + 1]) for index in range(4)]
        )
        self.encoder_tail = ConditionalResBlock(channels[-1], channels[-1], config.class_embed_dim)
        self.mu_head = nn.Conv2d(channels[-1], config.latent_channels, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(channels[-1], config.latent_channels, kernel_size=3, padding=1)

        self.decoder_stem = nn.Conv2d(config.latent_channels, channels[-1], kernel_size=3, padding=1)
        self.decoder_tail = ConditionalResBlock(channels[-1], channels[-1], config.class_embed_dim)
        reversed_channels = list(reversed(channels))
        self.upsamples = nn.ModuleList(
            [Upsample(reversed_channels[index], reversed_channels[index + 1]) for index in range(4)]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                ConditionalResBlock(reversed_channels[index + 1], reversed_channels[index + 1], config.class_embed_dim)
                for index in range(4)
            ]
        )
        self.output_norm = nn.GroupNorm(_group_count(b), b)
        self.output_conv = nn.Conv2d(b, config.input_channels, kernel_size=3, padding=1)

    def _validate_labels(self, labels: torch.Tensor) -> None:
        if labels.ndim != 1:
            raise ValueError(f"Expected one class label per sample, got {tuple(labels.shape)}")
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= self.config.num_classes):
            raise ValueError(f"Labels must be in [0, {self.config.num_classes - 1}]")

    def encode(self, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_labels(labels)
        expected = (self.config.input_channels, self.config.image_height, self.config.image_width)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(f"Expected inputs [batch, {expected}], got {tuple(inputs.shape)}")
        condition = self.class_embedding(labels)
        hidden = self.encoder_stem(inputs)
        for index, block in enumerate(self.encoder_blocks):
            hidden = block(hidden, condition)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        hidden = self.encoder_tail(hidden, condition)
        return self.mu_head(hidden), self.logvar_head(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate_labels(labels)
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.config.latent_shape:
            raise ValueError(
                f"Expected latent [batch, {self.config.latent_shape}], got {tuple(latent.shape)}"
            )
        condition = self.class_embedding(labels)
        hidden = self.decoder_tail(self.decoder_stem(latent), condition)
        for upsample, block in zip(self.upsamples, self.decoder_blocks):
            hidden = block(upsample(hidden), condition)
        output = self.output_conv(F.silu(self.output_norm(hidden)))
        expected = (latent.shape[0], self.config.input_channels, self.config.image_height, self.config.image_width)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"Expected decoder output {expected}, got {tuple(output.shape)}")
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_latent: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(inputs, labels)
        latent = self.reparameterize(mu, logvar) if sample_latent else mu
        return self.decode(latent, labels), mu, logvar


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
