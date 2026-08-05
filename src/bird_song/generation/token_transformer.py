from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TokenTransformerConfig:
    token_grid: int = 8
    codebook_size: int = 512
    num_classes: int = 3
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    feedforward_dim: int = 1_024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if min(self.token_grid, self.codebook_size, self.num_classes, self.d_model) < 1:
            raise ValueError("token transformer dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_layers < 1 or self.feedforward_dim < self.d_model:
            raise ValueError("invalid transformer depth or feedforward dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def token_count(self) -> int:
        return self.token_grid * self.token_grid

    @property
    def bos_token(self) -> int:
        return self.codebook_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TokenTransformerConfig":
        return cls(**values)


class ConditionalTokenTransformer(nn.Module):
    """Autoregressively model VQGAN codebook indices, not raw pixels."""

    def __init__(self, config: TokenTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.codebook_size + 1, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.token_count + 1, config.d_model))
        self.class_embedding = nn.Embedding(config.num_classes, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.codebook_size)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_embedding.weight, std=0.02)

    def _validate(self, tokens: torch.Tensor, labels: torch.Tensor) -> None:
        if tokens.ndim != 2 or tokens.shape[1] > self.config.token_count:
            raise ValueError(f"Expected token sequences with length <= {self.config.token_count}")
        if labels.ndim != 1 or labels.shape[0] != tokens.shape[0]:
            raise ValueError("token and label batch sizes must match")
        if tokens.numel() and (int(tokens.min()) < 0 or int(tokens.max()) >= self.config.codebook_size + 1):
            raise ValueError("token values are outside the codebook/BOS range")
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= self.config.num_classes):
            raise ValueError("labels are outside the configured class range")

    def forward(self, input_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate(input_tokens, labels)
        tokens = self.token_embedding(input_tokens)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = tokens + self.class_embedding(labels).unsqueeze(1)
        tokens = self.dropout(tokens)
        causal = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device), diagonal=1)
        encoded = self.transformer(tokens, mask=causal)
        return self.output(self.output_norm(encoded))

    def loss(self, token_grid: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if token_grid.ndim != 3 or tuple(token_grid.shape[1:]) != (self.config.token_grid, self.config.token_grid):
            raise ValueError(f"Expected token grids shaped (batch, {self.config.token_grid}, {self.config.token_grid})")
        flat = token_grid.flatten(1).long()
        bos = torch.full((flat.shape[0], 1), self.config.bos_token, dtype=torch.long, device=flat.device)
        logits = self(torch.cat((bos, flat[:, :-1]), dim=1), labels)
        return F.cross_entropy(logits.reshape(-1, self.config.codebook_size), flat.reshape(-1))

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        tokens = torch.full((labels.shape[0], 1), self.config.bos_token, dtype=torch.long, device=labels.device)
        for _ in range(self.config.token_count):
            logits = self(tokens, labels)[:, -1]
            if temperature == 0:
                next_token = logits.argmax(dim=-1)
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            tokens = torch.cat((tokens, next_token.unsqueeze(1)), dim=1)
        return tokens[:, 1:].view(-1, self.config.token_grid, self.config.token_grid)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
