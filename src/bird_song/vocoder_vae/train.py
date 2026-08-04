from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch import nn

from bird_song.runtime import atomic_torch_save, choose_device, save_json, seed_everything
from bird_song.vocoder import VocoderMelNormalizer, VocoderSpectrogramConfig
from bird_song.vocoder_data import VocoderSpectrogramDataset, make_vocoder_loader
from bird_song.vocoder_vae.losses import VocoderVAELossConfig, vae_loss
from bird_song.vocoder_vae.model import (
    ConditionalVocoderVAE,
    VocoderVAEConfig,
    count_trainable_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METRICS = ("loss", "recon", "mse", "mae", "multiscale", "time_grad", "frequency_grad", "kl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 80 x 256 spatial conditional vocoder VAE.")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/vocoder_spectrograms/vocoder_spectrogram_manifest.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/vocoder_spectrograms",
    )
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=PROJECT_ROOT / "artifacts/vocoder_spectrograms/normalization_stats.json",
    )
    parser.add_argument(
        "--vocoder-config",
        type=Path,
        default=PROJECT_ROOT / "configs/vocoder_spectrogram.json",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs/vocoder_vae.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/vocoder_vae")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke-batches",
        type=int,
        default=None,
        help="Limit each split to this many batches for a wiring smoke test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def beta_for_epoch(epoch: int, beta: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / warmup_epochs)


def run_epoch(
    model: ConditionalVocoderVAE,
    loader: Iterable,
    device: torch.device,
    loss_config: VocoderVAELossConfig,
    beta: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float = 5.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {metric: 0.0 for metric in METRICS}
    sample_count = 0
    for batch_index, (spectrograms, labels, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        spectrograms = spectrograms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                reconstruction, mu, logvar = model(
                    spectrograms,
                    labels,
                    sample_latent=training,
                )
                loss, parts = vae_loss(
                    reconstruction,
                    spectrograms,
                    mu,
                    logvar,
                    beta,
                    loss_config,
                )
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        batch_size = labels.numel()
        for metric in METRICS:
            totals[metric] += float(parts[metric].detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("No samples were processed")
    return {metric: totals[metric] / sample_count for metric in METRICS}


def main() -> None:
    args = parse_args()
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "gradient_clip": args.gradient_clip,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if args.epochs > 60:
        raise ValueError("The gated vocoder VAE cycle has a hard maximum of 60 epochs")
    if args.workers < 0 or args.beta_kl < 0 or args.kl_warmup_epochs < 0 or args.min_delta < 0:
        raise ValueError("workers, beta, warmup, and min-delta cannot be negative")
    if args.smoke_batches is not None and args.smoke_batches < 1:
        raise ValueError("--smoke-batches must be positive")

    existing = [args.output_dir / name for name in ("best.pt", "history.csv", "config.json")]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {[path.name for path in existing]}; choose another output or pass --overwrite"
        )

    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    cache_manifest = pd.read_csv(args.cache_manifest)
    classes = tuple(sorted(cache_manifest["name"].unique()))
    vocoder_config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    normalizer = VocoderMelNormalizer.from_json(args.normalizer)
    model_config = VocoderVAEConfig.from_json(args.model_config)
    if model_config.num_classes != len(classes):
        raise ValueError(
            f"Model expects {model_config.num_classes} classes, cache contains {len(classes)}"
        )
    if (model_config.image_height, model_config.image_width) != (
        vocoder_config.n_mels,
        vocoder_config.expected_frames,
    ):
        raise ValueError("VAE image shape does not match the vocoder mel contract")

    train_set = VocoderSpectrogramDataset(
        args.cache_manifest, args.cache_root, "train", classes, vocoder_config, normalizer
    )
    validation_set = VocoderSpectrogramDataset(
        args.cache_manifest, args.cache_root, "validation", classes, vocoder_config, normalizer
    )
    test_set = VocoderSpectrogramDataset(
        args.cache_manifest, args.cache_root, "test", classes, vocoder_config, normalizer
    )
    train_loader = make_vocoder_loader(
        train_set, args.batch_size, args.workers, training=True, seed=args.seed
    )
    validation_loader = make_vocoder_loader(validation_set, args.batch_size, args.workers)
    test_loader = make_vocoder_loader(test_set, args.batch_size, args.workers)

    model = ConditionalVocoderVAE(model_config).to(device)
    parameter_count = count_trainable_parameters(model)
    loss_config = VocoderVAELossConfig()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "classes": classes,
            "vocoder": vocoder_config.to_dict(),
            "normalizer": normalizer.to_dict(),
            "model": model_config.to_dict(),
            "loss": loss_config.to_dict(),
            "trainable_parameters": parameter_count,
            "device_used": str(device),
        },
    )

    history_path = args.output_dir / "history.csv"
    best_path = args.output_dir / "best.pt"
    best_validation_loss = float("inf")
    stale_epochs = 0
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        fieldnames = ["epoch", "beta", "seconds"] + [
            f"{split}_{metric}" for split in ("train", "validation") for metric in METRICS
        ]
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            beta = beta_for_epoch(epoch, args.beta_kl, args.kl_warmup_epochs)
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                loss_config,
                beta,
                optimizer=optimizer,
                scaler=scaler,
                gradient_clip=args.gradient_clip,
                max_batches=args.smoke_batches,
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                loss_config,
                beta,
                max_batches=args.smoke_batches,
            )
            row: dict[str, float | int] = {
                "epoch": epoch,
                "beta": beta,
                "seconds": time.perf_counter() - started,
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
            writer.writerow(row)
            history_file.flush()
            print(
                f"epoch={epoch:03d} beta={beta:.6f} train={train_metrics['loss']:.4f} "
                f"validation={validation_metrics['loss']:.4f} seconds={row['seconds']:.1f}"
            )
            if validation_metrics["loss"] < best_validation_loss - args.min_delta:
                best_validation_loss = validation_metrics["loss"]
                stale_epochs = 0
                atomic_torch_save(
                    {
                        "format_version": 1,
                        "model_type": "spatial_conditional_vocoder_vae",
                        "model_state": model.state_dict(),
                        "model_config": model_config.to_dict(),
                        "vocoder_config": vocoder_config.to_dict(),
                        "normalizer": normalizer.to_dict(),
                        "loss_config": loss_config.to_dict(),
                        "classes": list(classes),
                        "epoch": epoch,
                        "best_validation_loss": best_validation_loss,
                        "trainable_parameters": parameter_count,
                        "seed": args.seed,
                        "beta_kl": args.beta_kl,
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        loss_config,
        args.beta_kl,
        max_batches=args.smoke_batches,
    )
    save_json(
        args.output_dir / "test_metrics.json",
        {
            "checkpoint_epoch": checkpoint["epoch"],
            "metrics": test_metrics,
            "smoke_batches": args.smoke_batches,
        },
    )
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Test loss: {test_metrics['loss']:.6f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
