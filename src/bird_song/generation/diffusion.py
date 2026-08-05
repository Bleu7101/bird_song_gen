from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentDiffusionConfig:
    latent_channels: int = 128
    latent_grid: int = 8
    num_classes: int = 3
    time_embedding_dim: int = 128
    base_channels: int = 256
    diffusion_steps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        if min(self.latent_channels, self.latent_grid, self.num_classes, self.time_embedding_dim, self.base_channels) < 1:
            raise ValueError("latent diffusion dimensions must be positive")
        if self.diffusion_steps < 2 or not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("diffusion schedule is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LatentDiffusionConfig":
        return cls(**values)


def _sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -torch.log(torch.tensor(10_000.0, device=timesteps.device))
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class _ConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, bias = self.condition(condition).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        return x + self.conv2(F.silu(h))


class ConditionalLatentDenoiser(nn.Module):
    """Small class-conditional DDPM denoiser operating on VQGAN latents."""

    def __init__(self, config: LatentDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        condition_dim = config.time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, condition_dim),
            nn.SiLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
        )
        self.class_embedding = nn.Embedding(config.num_classes, condition_dim)
        self.input = nn.Conv2d(config.latent_channels, config.base_channels, 3, padding=1)
        self.blocks = nn.ModuleList([_ConditionedResidualBlock(config.base_channels, condition_dim) for _ in range(4)])
        self.output = nn.Sequential(
            nn.GroupNorm(32, config.base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.base_channels, config.latent_channels, 3, padding=1),
        )

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        expected = (self.config.latent_channels, self.config.latent_grid, self.config.latent_grid)
        if latents.ndim != 4 or tuple(latents.shape[1:]) != expected:
            raise ValueError(f"Expected latents shaped (batch, {expected}), got {tuple(latents.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != latents.shape[0]:
            raise ValueError("timesteps must have one value per latent")
        if labels.ndim != 1 or labels.shape[0] != latents.shape[0]:
            raise ValueError("labels must have one value per latent")
        condition = self.time_mlp(_sinusoidal_embedding(timesteps, self.config.time_embedding_dim)) + self.class_embedding(labels)
        h = self.input(latents)
        for block in self.blocks:
            h = block(h, condition)
        return self.output(h)


def diffusion_schedule(config: LatentDiffusionConfig, device: torch.device) -> tuple[torch.Tensor, ...]:
    betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = alpha_bars[timesteps].view(-1, 1, 1, 1)
    return alpha_bar.sqrt() * latents + (1.0 - alpha_bar).sqrt() * noise


def diffusion_loss(
    model: ConditionalLatentDenoiser,
    latents: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    betas, _, alpha_bars = diffusion_schedule(model.config, latents.device)
    del betas
    timesteps = torch.randint(
        model.config.diffusion_steps,
        (latents.shape[0],),
        device=latents.device,
        generator=generator,
    )
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    noisy = q_sample(latents, timesteps, noise, alpha_bars)
    return F.mse_loss(model(noisy, timesteps, labels), noise)


@torch.no_grad()
def sample_latents(
    model: ConditionalLatentDenoiser,
    labels: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Ancestral DDPM sampling; use latent diffusion before decoding through VQGAN."""
    config = model.config
    betas, alphas, alpha_bars = diffusion_schedule(config, labels.device)
    latents = torch.randn(
        labels.shape[0],
        config.latent_channels,
        config.latent_grid,
        config.latent_grid,
        device=labels.device,
        dtype=model.input.weight.dtype,
        generator=generator,
    )
    for step in range(config.diffusion_steps - 1, -1, -1):
        timestep = torch.full((labels.shape[0],), step, device=labels.device, dtype=torch.long)
        predicted_noise = model(latents, timestep, labels)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        mean = (latents - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if step:
            noise = torch.randn(latents.shape, device=labels.device, dtype=latents.dtype, generator=generator)
            latents = mean + betas[step].sqrt() * noise
        else:
            latents = mean
    return latents


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
