from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VocoderVAELossConfig:
    event_weight: float = 1.5
    multiscale_weight: float = 0.25
    time_gradient_weight: float = 0.5
    frequency_gradient_weight: float = 0.5

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Loss weights cannot be negative")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def first_difference(inputs: torch.Tensor, dimension: int) -> torch.Tensor:
    if dimension == -1:
        return inputs[..., 1:] - inputs[..., :-1]
    if dimension == -2:
        return inputs[..., 1:, :] - inputs[..., :-1, :]
    raise ValueError("Only time (-1) and frequency (-2) differences are supported")


def detail_aware_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    config: VocoderVAELossConfig,
) -> dict[str, torch.Tensor]:
    absolute_error = (reconstruction - target).abs()
    event_weights = 1.0 + config.event_weight * torch.sigmoid(target)
    weighted_l1 = (absolute_error * event_weights).sum() / event_weights.sum()
    pooled = [
        F.l1_loss(F.avg_pool2d(reconstruction, scale), F.avg_pool2d(target, scale))
        for scale in (2, 4)
    ]
    multiscale = torch.stack(pooled).mean()
    reconstruction_dt = first_difference(reconstruction, -1)
    target_dt = first_difference(target, -1)
    reconstruction_df = first_difference(reconstruction, -2)
    target_df = first_difference(target, -2)
    time_grad = ((reconstruction_dt - target_dt).abs() * (1.0 + target_dt.abs().detach())).mean()
    frequency_grad = ((reconstruction_df - target_df).abs() * (1.0 + target_df.abs().detach())).mean()
    objective = (
        weighted_l1
        + config.multiscale_weight * multiscale
        + config.time_gradient_weight * time_grad
        + config.frequency_gradient_weight * frequency_grad
    )
    return {
        "recon": objective,
        "mse": F.mse_loss(reconstruction, target),
        "mae": absolute_error.mean(),
        "multiscale": multiscale,
        "time_grad": time_grad,
        "frequency_grad": frequency_grad,
    }


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    config: VocoderVAELossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if beta < 0:
        raise ValueError("beta cannot be negative")
    parts = detail_aware_reconstruction_loss(reconstruction, target, config)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    loss = parts["recon"] + beta * kl
    return loss, {"loss": loss, **parts, "kl": kl}
