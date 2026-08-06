from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerGeneratorConfig:
    """Architecture contract for rectangular continuous BigVGAN mel generation."""

    height: int = 80
    width: int = 256
    patch_height: int = 16
    patch_width: int = 16
    channels: int = 1
    num_classes: int = 3
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    feedforward_dim: int = 1_024
    dropout: float = 0.10
    min_log_scale: float = -5.0
    max_log_scale: float = 1.0

    def __post_init__(self) -> None:
        dimensions = (self.height, self.width, self.patch_height, self.patch_width)
        if any(value < 1 for value in dimensions):
            raise ValueError("height, width, and patch dimensions must be positive")
        if self.height % self.patch_height or self.width % self.patch_width:
            raise ValueError("height and width must be divisible by their patch dimensions")
        if self.channels < 1 or self.num_classes < 1:
            raise ValueError("channels and num_classes must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_layers < 1 or self.feedforward_dim < self.d_model:
            raise ValueError("num_layers must be positive and feedforward_dim must be at least d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.min_log_scale >= self.max_log_scale:
            raise ValueError("min_log_scale must be less than max_log_scale")

    @property
    def grid_height(self) -> int:
        return self.height // self.patch_height

    @property
    def grid_width(self) -> int:
        return self.width // self.patch_width

    @property
    def patch_count(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def patch_dimension(self) -> int:
        return self.channels * self.patch_height * self.patch_width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TransformerGeneratorConfig":
        # Accept the old square schema when inspecting a legacy checkpoint, but
        # always emit the explicit rectangular schema for new checkpoints.
        normalized = dict(values)
        if "image_size" in normalized:
            image_size = int(normalized.pop("image_size"))
            normalized.setdefault("height", image_size)
            normalized.setdefault("width", image_size)
        if "patch_size" in normalized:
            patch_size = int(normalized.pop("patch_size"))
            normalized.setdefault("patch_height", patch_size)
            normalized.setdefault("patch_width", patch_size)
        return cls(**normalized)

    @classmethod
    def from_json(cls, path: Path) -> "TransformerGeneratorConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class ConditionalSpectrogramTransformer(nn.Module):
    """Generate normalized log-mel arrays one time-major patch at a time."""

    def __init__(self, config: TransformerGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_projection = nn.Linear(config.patch_dimension, config.d_model)
        self.beginning_of_image = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.patch_count, config.d_model))
        self.species_embedding = nn.Embedding(config.num_classes, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.distribution_head = nn.Linear(config.d_model, config.patch_dimension * 2)

        nn.init.trunc_normal_(self.beginning_of_image, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.species_embedding.weight, std=0.02)

    def _validate_labels(self, labels: torch.Tensor) -> None:
        if labels.ndim != 1:
            raise ValueError(f"Expected one label per sample, got shape {tuple(labels.shape)}")
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= self.config.num_classes):
            raise ValueError(f"Labels must be in [0, {self.config.num_classes - 1}]")

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert [B,C,H,W] arrays to time-major then frequency-major patches."""

        expected = (self.config.channels, self.config.height, self.config.width)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected inputs shaped (batch, {expected}), got {tuple(images.shape)}")
        batch_size = images.shape[0]
        return (
            images.reshape(
                batch_size,
                self.config.channels,
                self.config.grid_height,
                self.config.patch_height,
                self.config.grid_width,
                self.config.patch_width,
            )
            .permute(0, 4, 2, 1, 3, 5)
            .reshape(batch_size, self.config.patch_count, self.config.patch_dimension)
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        expected = (self.config.patch_count, self.config.patch_dimension)
        if patches.ndim != 3 or tuple(patches.shape[1:]) != expected:
            raise ValueError(f"Expected patches shaped (batch, {expected}), got {tuple(patches.shape)}")
        batch_size = patches.shape[0]
        return (
            patches.reshape(
                batch_size,
                self.config.grid_width,
                self.config.grid_height,
                self.config.channels,
                self.config.patch_height,
                self.config.patch_width,
            )
            .permute(0, 3, 2, 4, 1, 5)
            .reshape(batch_size, self.config.channels, self.config.height, self.config.width)
        )

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _encode_inputs(self, previous_patches: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate_labels(labels)
        batch_size = labels.shape[0]
        if previous_patches.shape[0] != batch_size:
            raise ValueError("Patch and label batch sizes do not match")
        if previous_patches.shape[1] >= self.config.patch_count:
            raise ValueError("previous_patches must contain fewer than patch_count patches")

        beginning = self.beginning_of_image.expand(batch_size, -1, -1)
        embedded_previous = self.patch_projection(previous_patches)
        tokens = torch.cat((beginning, embedded_previous), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = tokens + self.species_embedding(labels).unsqueeze(1)
        return self.embedding_dropout(tokens)

    def _distribution_from_previous(
        self,
        previous_patches: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._encode_inputs(previous_patches, labels)
        encoded = self.transformer(tokens, mask=self.causal_mask(tokens.shape[1], tokens.device))
        parameters = self.distribution_head(self.output_norm(encoded))
        raw_mean, raw_log_scale = parameters.chunk(2, dim=-1)
        mean = raw_mean.tanh()
        log_scale = raw_log_scale.clamp(self.config.min_log_scale, self.config.max_log_scale)
        return mean, log_scale

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher-forced distributions for every target patch."""

        patches = self.patchify(images)
        return self._distribution_from_previous(patches[:, :-1], labels)

    @torch.inference_mode()
    def generate(
        self,
        labels: torch.Tensor,
        temperature: float = 0.8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample normalized log-mel arrays conditioned on integer species labels."""

        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        self._validate_labels(labels)
        was_training = self.training
        self.eval()
        generated = torch.empty(
            labels.shape[0],
            0,
            self.config.patch_dimension,
            dtype=self.position_embedding.dtype,
            device=labels.device,
        )
        for _ in range(self.config.patch_count):
            mean, log_scale = self._distribution_from_previous(generated, labels)
            next_mean = mean[:, -1]
            if temperature == 0:
                next_patch = next_mean
            else:
                noise = torch.randn(
                    next_mean.shape,
                    dtype=next_mean.dtype,
                    device=next_mean.device,
                    generator=generator,
                )
                next_patch = next_mean + temperature * log_scale[:, -1].exp() * noise
            generated = torch.cat((generated, next_patch.clamp(-1.0, 1.0).unsqueeze(1)), dim=1)
        if was_training:
            self.train()
        return self.unpatchify(generated)

    def metadata(self) -> dict[str, Any]:
        return self.config.to_dict()


def gaussian_patch_nll(
    images: torch.Tensor,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    model: ConditionalSpectrogramTransformer,
) -> torch.Tensor:
    """Gaussian negative log likelihood for normalized continuous patches."""

    targets = model.patchify(images).float()
    mean = mean.float()
    log_scale = log_scale.float()
    inverse_scale = torch.exp(-log_scale)
    standardized = (targets - mean) * inverse_scale
    return (0.5 * standardized.square() + log_scale + 0.5 * math.log(2 * math.pi)).mean()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
