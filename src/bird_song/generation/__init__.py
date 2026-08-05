"""Generator models for normalized log-mel spectrograms.

The classifier package is intentionally separate.  These models all operate on
the existing ``[batch, 1, 128, 128]`` spectrogram contract and can therefore be
compared without changing the classifier or the published preprocessing.
"""

from .diffusion import (
    ConditionalLatentDenoiser,
    LatentDiffusionConfig,
    diffusion_loss,
    sample_latents,
)
from .audio_decode import normalized_logmel_to_waveform, write_waveform
from .evaluation import detail_metrics, diversity_metrics
from .token_transformer import ConditionalTokenTransformer, TokenTransformerConfig
from .vqgan import ConditionalVQGAN, VQGANConfig
from .wgan_gp import (
    ConditionalCritic,
    ConditionalGenerator,
    WGANConfig,
    gradient_penalty,
)

__all__ = [
    "ConditionalCritic",
    "ConditionalGenerator",
    "ConditionalLatentDenoiser",
    "ConditionalTokenTransformer",
    "ConditionalVQGAN",
    "LatentDiffusionConfig",
    "TokenTransformerConfig",
    "VQGANConfig",
    "WGANConfig",
    "detail_metrics",
    "diversity_metrics",
    "diffusion_loss",
    "gradient_penalty",
    "normalized_logmel_to_waveform",
    "sample_latents",
    "write_waveform",
]
