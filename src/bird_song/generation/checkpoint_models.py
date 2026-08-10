"""Inference-only model definitions for the portable VAE-v3/diffusion checkpoints.

The original training implementations live in notebooks and on the dedicated
diffusion branch.  This module deliberately contains only the architecture and
sampling contracts needed to evaluate already-trained checkpoints; it has no
optimizer, loss, or training code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


GENERATOR_CLASSES = (
    "Northern Cardinal",
    "Song Sparrow",
    "American Robin",
)

VAE_PARAMETER_COUNT = 5_365_025
DIFFUSION_PARAMETER_COUNT = 18_443_841
VAE_TEMPERATURE = 0.35
DIFFUSION_TIMESTEPS = 1_000
DIFFUSION_DDIM_STEPS = 100
DIFFUSION_DDIM_ETA = 0.0
DIFFUSION_GUIDANCE = 3.0
DIFFUSION_CLAMP = 4.0
NORMALIZATION_MEAN = -51.5400096102764
NORMALIZATION_STD = 14.894513218933453


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionalResBlock(nn.Module):
    """VAE-v3 residual block with FiLM class conditioning."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
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
        # These initializers are part of the checkpoint architecture contract.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        scale, shift = self.film(condition).chunk(2, dim=1)
        h = h * (1.0 + 0.1 * torch.tanh(scale[:, :, None, None]))
        h = h + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return residual + h


class VAEDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VAEUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ConditionalVAE(nn.Module):
    """The VAE-v3 decoder and encoder needed for posterior-bank sampling."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 128,
        input_channels: int = 1,
        latent_channels: int = 16,
        latent_size: int = 16,
        class_embed_dim: int = 64,
        base_channels: int = 32,
    ):
        super().__init__()
        ratio = image_size // latent_size
        if image_size % latent_size or ratio < 2 or ratio & (ratio - 1):
            raise ValueError("image_size / latent_size must be a power of two >= 2")
        num_downsamples = int(math.log2(ratio))
        if num_downsamples != 3:
            raise ValueError("V3 expects 128x128 inputs and a 16x16 latent map.")

        self.num_classes = num_classes
        self.image_size = image_size
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.latent_dim = latent_channels * latent_size * latent_size
        self.class_embed_dim = class_embed_dim
        self.base_channels = base_channels

        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        channels = [base_channels * multiplier for multiplier in (1, 2, 4, 8)]
        self.encoder_stem = nn.Conv2d(input_channels, channels[0], 3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            [ConditionalResBlock(c, c, class_embed_dim) for c in channels]
        )
        self.downsamples = nn.ModuleList(
            [VAEDownsample(channels[i], channels[i + 1]) for i in range(num_downsamples)]
        )
        self.encoder_tail = ConditionalResBlock(channels[-1], channels[-1], class_embed_dim)
        self.mu_head = nn.Conv2d(channels[-1], latent_channels, 3, padding=1)
        self.logvar_head = nn.Conv2d(channels[-1], latent_channels, 3, padding=1)

        self.decoder_stem = nn.Conv2d(latent_channels, channels[-1], 3, padding=1)
        self.decoder_tail = ConditionalResBlock(channels[-1], channels[-1], class_embed_dim)
        reversed_channels = list(reversed(channels))
        self.upsamples = nn.ModuleList(
            [VAEUpsample(reversed_channels[i], reversed_channels[i + 1]) for i in range(num_downsamples)]
        )
        self.decoder_blocks = nn.ModuleList(
            [ConditionalResBlock(c, c, class_embed_dim) for c in reversed_channels[1:]]
        )
        self.output_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.output_conv = nn.Conv2d(channels[0], input_channels, 3, padding=1)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.class_embedding(labels)
        h = self.encoder_stem(x)
        for index, block in enumerate(self.encoder_blocks):
            h = block(h, condition)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)
        h = self.encoder_tail(h, condition)
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
        return mu + epsilon * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = self.class_embedding(labels)
        h = self.decoder_tail(self.decoder_stem(z), condition)
        for upsample, block in zip(self.upsamples, self.decoder_blocks):
            h = block(upsample(h), condition)
        return self.output_conv(F.silu(self.output_norm(h)))

    def forward(self, x: torch.Tensor, labels: torch.Tensor, sample_latent: bool = True):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        return self.decode(z, labels), mu, logvar


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / (half - 1))
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class DiffusionAttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        attention = torch.softmax(q @ k / math.sqrt(channels), dim=-1)
        v = v.reshape(batch, channels, height * width).permute(0, 2, 1)
        output = (attention @ v).permute(0, 2, 1).reshape(batch, channels, height, width)
        return x + self.proj(output)


class DiffusionDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DiffusionUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    """The exact conditional U-Net used by the recorded diffusion checkpoint."""

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 64,
        dim_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_classes: int = 3,
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        image_size: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        dims = [base] + [base * multiplier for multiplier in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        resolution = image_size
        self.downs = nn.ModuleList()
        for index, (din, dout) in enumerate(in_out):
            is_last = index >= num_resolutions - 1
            use_attn = resolution in attn_resolutions
            blocks = [DiffusionResBlock(din, din, emb_dim, dropout) for _ in range(num_res_blocks)]
            self.downs.append(
                nn.ModuleList(
                    [
                        blocks[0],
                        blocks[1] if num_res_blocks > 1 else DiffusionResBlock(din, din, emb_dim, dropout),
                        DiffusionAttnBlock(din) if use_attn else nn.Identity(),
                        nn.Conv2d(din, dout, 3, padding=1) if is_last else DiffusionDownsample(din, dout),
                    ]
                )
            )
            if not is_last:
                resolution //= 2

        middle_dim = dims[-1]
        self.mid_block1 = DiffusionResBlock(middle_dim, middle_dim, emb_dim, dropout)
        self.mid_attn = DiffusionAttnBlock(middle_dim)
        self.mid_block2 = DiffusionResBlock(middle_dim, middle_dim, emb_dim, dropout)

        self.ups = nn.ModuleList()
        for index, (din, dout) in enumerate(reversed(in_out)):
            is_last = index >= num_resolutions - 1
            use_attn = resolution in attn_resolutions
            self.ups.append(
                nn.ModuleList(
                    [
                        DiffusionResBlock(dout + din, dout, emb_dim, dropout),
                        DiffusionResBlock(dout + din, dout, emb_dim, dropout),
                        DiffusionAttnBlock(dout) if use_attn else nn.Identity(),
                        nn.Conv2d(dout, din, 3, padding=1) if is_last else DiffusionUpsample(dout, din),
                    ]
                )
            )
            if not is_last:
                resolution *= 2

        self.final_res = DiffusionResBlock(base + base, base, emb_dim, dropout)
        self.final_norm = nn.GroupNorm(_group_count(base), base)
        self.final_conv = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t) + self.class_emb(labels)
        x = self.init_conv(x)
        residual = x
        skips: list[torch.Tensor] = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            skips.append(x)
            x = block2(x, emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, emb)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x, emb)
            x = attn(x)
            x = upsample(x)
        x = torch.cat([x, residual], dim=1)
        x = self.final_res(x, emb)
        return self.final_conv(F.silu(self.final_norm(x)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999).float()


def build_diffusion_schedule(device: torch.device, timesteps: int = DIFFUSION_TIMESTEPS) -> dict[str, torch.Tensor]:
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
    }


def _extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    output = values.gather(0, t)
    return output.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


@torch.inference_mode()
def ddim_sample(
    model: ConditionalUNet,
    labels: torch.Tensor,
    generator: torch.Generator,
    schedule: Mapping[str, torch.Tensor],
    steps: int = DIFFUSION_DDIM_STEPS,
    eta: float = DIFFUSION_DDIM_ETA,
    guidance: float = DIFFUSION_GUIDANCE,
    clamp_samples: float | None = DIFFUSION_CLAMP,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one or more examples with the recorded DDIM update.

    The caller passes a per-sample generator when deterministic resume is
    required.  The CLI intentionally calls this with a batch of one for every
    sample, so outer chunking cannot change the random stream or output.
    """
    model.eval()
    device = labels.device
    batch = len(labels)
    x = initial_noise if initial_noise is not None else torch.randn(
        (batch, 1, 128, 128), generator=generator, device=device
    )
    if tuple(x.shape) != (batch, 1, 128, 128):
        raise ValueError(f"initial_noise must have shape {(batch, 1, 128, 128)}, got {tuple(x.shape)}")
    x = x.to(device=device, dtype=torch.float32)
    seq = torch.linspace(0, DIFFUSION_TIMESTEPS - 1, int(steps), device=device).round().long().tolist()
    seq = list(reversed(seq))
    alpha_bar = schedule["alpha_bar"]
    for position, timestep in enumerate(seq):
        t = torch.full((batch,), timestep, device=device, dtype=torch.long)
        if guidance != 1.0:
            null_labels = torch.full_like(labels, model.num_classes)
            eps_cond = model(x, t, labels)
            eps_uncond = model(x, t, null_labels)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
        else:
            eps = model(x, t, labels)
        alpha_t = _extract(alpha_bar, t, x.shape)
        x0 = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        if clamp_samples is not None:
            x0 = x0.clamp(-clamp_samples, clamp_samples)
        next_timestep = seq[position + 1] if position + 1 < len(seq) else -1
        if next_timestep < 0:
            x = x0
            continue
        t_next = torch.full((batch,), next_timestep, device=device, dtype=torch.long)
        alpha_next = _extract(alpha_bar, t_next, x.shape)
        sigma = eta * (((1 - alpha_next) / (1 - alpha_t)) * (1 - alpha_t / alpha_next)).clamp(min=0).sqrt()
        noise = torch.randn(x.shape, generator=generator, device=device) if eta else torch.zeros_like(x)
        x = (
            alpha_next.sqrt() * x0
            + (1 - alpha_next - sigma**2).clamp(min=0).sqrt() * eps
            + sigma * noise
        )
    return x


def classifier_scale_from_standardized(
    standardized: torch.Tensor | Any,
    mean: float = NORMALIZATION_MEAN,
    std: float = NORMALIZATION_STD,
) -> torch.Tensor:
    """Convert the generator's global-standardized output to classifier scale."""
    tensor = standardized if isinstance(standardized, torch.Tensor) else torch.as_tensor(standardized)
    logmel_db = tensor * std + mean
    relative_db = logmel_db - logmel_db.amax(dim=(-2, -1), keepdim=True)
    relative_db = relative_db.clamp(-80.0, 0.0)
    return relative_db.mul(2.0 / 80.0).add(1.0).to(torch.float32)


def checkpoint_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _state_mapping(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get(key)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint is missing {key}")
    return state


def load_vae_model(path: Path, device: torch.device) -> tuple[ConditionalVAE, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    model = ConditionalVAE(
        num_classes=3,
        image_size=int(config.get("IMAGE_SIZE", 128)),
        input_channels=int(config.get("INPUT_CHANNELS", 1)),
        latent_channels=int(config.get("LATENT_CHANNELS", 16)),
        latent_size=int(config.get("LATENT_SIZE", 16)),
        class_embed_dim=int(config.get("CLASS_EMBED_DIM", 64)),
        base_channels=int(config.get("BASE_CHANNELS", 32)),
    ).to(device)
    expected = int(checkpoint.get("parameter_counts", {}).get("total", VAE_PARAMETER_COUNT))
    actual = checkpoint_parameter_count(model)
    if actual != expected or actual != VAE_PARAMETER_COUNT:
        raise ValueError(f"VAE parameter count mismatch: checkpoint={expected}, model={actual}")
    state = _state_mapping(checkpoint, "model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(checkpoint)


def load_diffusion_model(path: Path, device: torch.device) -> tuple[ConditionalUNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = dict(checkpoint.get("config", {}))
    model = ConditionalUNet(
        in_channels=int(config.get("INPUT_CHANNELS", 1)),
        base=int(config.get("BASE_CHANNELS", 64)),
        dim_mults=tuple(config.get("DIM_MULTS", (1, 2, 4, 4))),
        num_classes=3,
        num_res_blocks=int(config.get("NUM_RES_BLOCKS", 2)),
        attn_resolutions=tuple(config.get("ATTN_RESOLUTIONS", (16,))),
        image_size=int(config.get("IMAGE_SIZE", 128)),
        dropout=float(config.get("DROPOUT", 0.1)),
    ).to(device)
    expected = int(checkpoint.get("parameter_counts", {}).get("total", DIFFUSION_PARAMETER_COUNT))
    actual = checkpoint_parameter_count(model)
    if actual != expected or actual != DIFFUSION_PARAMETER_COUNT:
        raise ValueError(f"diffusion parameter count mismatch: checkpoint={expected}, model={actual}")
    state = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("diffusion checkpoint is missing ema_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(checkpoint)
