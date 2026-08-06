from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from bird_song.config import SpectrogramConfig
from bird_song.runtime import atomic_torch_save, choose_device, save_json, seed_everything
from bird_song.transformer.data import CachedSpectrogramDataset, make_cached_loader
from bird_song.transformer.model import (
    ConditionalSpectrogramTransformer,
    TransformerGeneratorConfig,
    count_trainable_parameters,
    gaussian_patch_nll,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a species-conditional autoregressive transformer on cached log-mel images."
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/spectrograms/spectrogram_manifest.csv",
    )
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/transformer.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/transformer_generator")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    raw_model: ConditionalSpectrogramTransformer,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = total_items = 0
    for spectrograms, labels, _ in loader:
        spectrograms = spectrograms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mean, log_scale = model(spectrograms, labels)
        loss = gaussian_patch_nll(spectrograms, mean, log_scale, raw_model)
        total_loss += float(loss) * labels.numel()
        total_items += labels.numel()
    return total_loss / total_items


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.min_delta < 0 or args.gradient_clip <= 0:
        raise ValueError("min_delta cannot be negative and gradient_clip must be positive")

    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    manifest = pd.read_csv(args.cache_manifest)
    classes = tuple(sorted(manifest["name"].unique()))
    spectrogram_config = SpectrogramConfig.from_json(args.spectrogram_config)
    model_config = TransformerGeneratorConfig.from_json(args.model_config)
    if len(classes) != model_config.num_classes:
        raise ValueError(
            f"Model config expects {model_config.num_classes} classes, but the manifest has {len(classes)}"
        )
    expected_shape = (spectrogram_config.n_mels, spectrogram_config.spectrogram_width)
    if expected_shape != (model_config.image_size, model_config.image_size):
        raise ValueError(
            f"Spectrogram config produces {expected_shape}, but transformer expects "
            f"{(model_config.image_size, model_config.image_size)}"
        )

    train_set = CachedSpectrogramDataset(
        args.cache_manifest,
        args.cache_root,
        "train",
        classes,
        model_config.image_size,
        specaugment=args.specaugment,
    )
    validation_set = CachedSpectrogramDataset(
        args.cache_manifest,
        args.cache_root,
        "validation",
        classes,
        model_config.image_size,
    )
    train_loader = make_cached_loader(
        train_set,
        args.batch_size,
        args.workers,
        training=True,
        seed=args.seed,
    )
    validation_loader = make_cached_loader(validation_set, args.batch_size, args.workers)

    raw_model = ConditionalSpectrogramTransformer(model_config).to(device)
    trainable_parameters = count_trainable_parameters(raw_model)
    model: nn.Module = raw_model
    if args.compile_model:
        model = torch.compile(model)

    existing = [args.output_dir / name for name in ("config.json", "history.csv", "best.pt")]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite files in {args.output_dir}: {names}. "
            "Choose another --output-dir or pass --overwrite."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "classes": classes,
            "model": model_config.to_dict(),
            "spectrogram": spectrogram_config.to_dict(),
            "trainable_parameters": trainable_parameters,
            "device_used": str(device),
        },
    )

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_validation_nll = float("inf")
    stale_epochs = 0
    history_path = args.output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["epoch", "train_nll", "validation_nll", "lr", "seconds"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            total_train_loss = total_train_items = 0
            for spectrograms, labels, _ in train_loader:
                spectrograms = spectrograms.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    mean, log_scale = model(spectrograms, labels)
                    loss = gaussian_patch_nll(spectrograms, mean, log_scale, raw_model)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(raw_model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                total_train_loss += float(loss.detach()) * labels.numel()
                total_train_items += labels.numel()

            validation_nll = evaluate(model, raw_model, validation_loader, device)
            row = {
                "epoch": epoch,
                "train_nll": total_train_loss / total_train_items,
                "validation_nll": validation_nll,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            history_file.flush()
            print(
                " ".join(
                    f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in row.items()
                )
            )

            if validation_nll < best_validation_nll - args.min_delta:
                best_validation_nll = validation_nll
                stale_epochs = 0
                atomic_torch_save(
                    {
                        "format_version": 1,
                        "model_type": "conditional_autoregressive_spectrogram_transformer",
                        "model_state": raw_model.state_dict(),
                        "model_config": model_config.to_dict(),
                        "classes": list(classes),
                        "spectrogram_config": spectrogram_config.to_dict(),
                        "trainable_parameters": trainable_parameters,
                        "seed": args.seed,
                        "epoch": epoch,
                        "best_validation_nll": best_validation_nll,
                    },
                    args.output_dir / "best.pt",
                )
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    print(f"Best validation NLL: {best_validation_nll:.4f}")
    print(f"Checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
