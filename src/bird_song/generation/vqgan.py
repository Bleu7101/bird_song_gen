from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .wgan_gp import _validate_labels


@dataclass(frozen=True)
class VQGANConfig:
    """Compact adversarial tokenizer for the shared 128 x 128 representation."""

    image_size: int = 128
    channels: int = 1
    num_classes: int = 3
    base_channels: int = 64
    latent_channels: int = 128
    codebook_size: int = 512
    commitment_weight: float = 0.25
    label_dim: int = 64
    discriminator_base_channels: int = 64

    def __post_init__(self) -> None:
        if self.image_size != 128:
            raise ValueError("The shared generator branch currently requires image_size=128")
        if min(self.channels, self.num_classes, self.base_channels, self.latent_channels) < 1:
            raise ValueError("model dimensions must be positive")
        if self.codebook_size < 2 or self.commitment_weight < 0 or self.label_dim < 1:
            raise ValueError("codebook_size, commitment_weight, and label_dim must be valid")

    @property
    def latent_grid(self) -> int:
        return self.image_size // 16

    @property
    def token_count(self) -> int:
        return self.latent_grid**2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VQGANConfig":
        return cls(**values)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, commitment_weight: float) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latents.ndim != 4 or latents.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected latents shaped (batch, {self.embedding_dim}, height, width), got {tuple(latents.shape)}"
            )
        flat = latents.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embedding.weight.t()
            + self.embedding.weight.square().sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(latents.shape[0], latents.shape[2], latents.shape[3], -1)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        commitment = F.mse_loss(latents, quantized.detach())
        codebook = F.mse_loss(quantized, latents.detach())
        loss = codebook + self.commitment_weight * commitment
        # Straight-through estimator: encoder sees quantized gradients.
        quantized = latents + (quantized - latents).detach()
        indices = indices.view(latents.shape[0], latents.shape[2], latents.shape[3])
        return quantized, indices, loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError(f"Expected token grid shaped (batch, height, width), got {tuple(indices.shape)}")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.codebook_size):
            raise ValueError("token indices are outside the codebook")
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual and in_channels == out_channels
        self.block = nn.Sequential(
            nn.GroupNorm(max(1, min(32, in_channels // 8)), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, min(32, out_channels // 8)), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return y + x if self.residual else y


class SpectrogramEncoder(nn.Module):
    def __init__(self, config: VQGANConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.input = nn.Conv2d(config.channels, c, 3, padding=1)
        self.blocks = nn.ModuleList()
        channels = c
        for multiplier in (2, 4, 8, 16):
            self.blocks.append(_ConvBlock(channels, channels))
            self.blocks.append(nn.Conv2d(channels, c * multiplier, 4, stride=2, padding=1))
            channels = c * multiplier
        self.blocks.extend((_ConvBlock(channels, channels), nn.Conv2d(channels, config.latent_channels, 3, padding=1)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.input(images)
        for block in self.blocks:
            x = block(x)
        return x


class SpectrogramDecoder(nn.Module):
    def __init__(self, config: VQGANConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.label_embedding = nn.Embedding(config.num_classes, config.label_dim)
        self.label_projection = nn.Linear(config.label_dim, config.latent_channels)
        self.input = nn.Conv2d(config.latent_channels, c * 16, 3, padding=1)
        self.blocks = nn.ModuleList()
        channels = c * 16
        for next_channels in (c * 8, c * 4, c * 2, c):
            self.blocks.extend((_ConvBlock(channels, channels), nn.ConvTranspose2d(channels, next_channels, 4, stride=2, padding=1)))
            channels = next_channels
        self.blocks.append(_ConvBlock(channels, channels))
        self.output = nn.Sequential(
            nn.GroupNorm(max(1, min(32, channels // 8)), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, config.channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _validate_labels(labels, latents.shape[0], self.label_embedding.num_embeddings)
        condition = self.label_projection(self.label_embedding(labels)).unsqueeze(-1).unsqueeze(-1)
        x = self.input(latents + condition)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class ConditionalVQGAN(nn.Module):
    """VQGAN-style tokenizer/decoder; the discriminator is kept separate."""

    def __init__(self, config: VQGANConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SpectrogramEncoder(config)
        self.quantizer = VectorQuantizer(config.codebook_size, config.latent_channels, config.commitment_weight)
        self.decoder = SpectrogramDecoder(config)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(self.encoder(images))

    def decode(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.lookup(indices), labels)

    def decode_latents(self, latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Decode continuous latents, used by latent diffusion samples."""
        expected = (self.config.latent_channels, self.config.latent_grid, self.config.latent_grid)
        if latents.ndim != 4 or tuple(latents.shape[1:]) != expected:
            raise ValueError(f"Expected latents shaped (batch, {expected}), got {tuple(latents.shape)}")
        return self.decoder(latents, labels)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized, indices, vq_loss = self.encode(images)
        reconstruction = self.decoder(quantized, labels)
        return reconstruction, indices, quantized, vq_loss


class PatchDiscriminator(nn.Module):
    """Local critic that encourages onsets and harmonic ridges to remain sharp."""

    def __init__(self, config: VQGANConfig) -> None:
        super().__init__()
        c = config.discriminator_base_channels
        layers: list[nn.Module] = [nn.Conv2d(config.channels, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        channels = c
        for multiplier in (2, 4, 8):
            layers.extend((nn.Conv2d(channels, c * multiplier, 4, stride=2, padding=1), nn.GroupNorm(8, c * multiplier), nn.LeakyReLU(0.2, inplace=True)))
            channels = c * multiplier
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def reconstruction_loss(real: torch.Tensor, reconstruction: torch.Tensor, gradient_weight: float = 0.5) -> torch.Tensor:
    l1 = F.l1_loss(reconstruction, real)
    time_gradient = F.l1_loss(reconstruction[..., 1:] - reconstruction[..., :-1], real[..., 1:] - real[..., :-1])
    frequency_gradient = F.l1_loss(reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :], real[:, :, 1:, :] - real[:, :, :-1, :])
    return l1 + gradient_weight * (time_gradient + frequency_gradient)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
