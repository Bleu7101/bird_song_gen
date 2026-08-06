from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from bird_song import DEFAULT_CLASSES
from bird_song.runtime import atomic_torch_save, choose_device, save_json, seed_everything
from bird_song.transformer.data import make_cached_loader
from bird_song.transformer.model import (
    ConditionalSpectrogramTransformer,
    TransformerGeneratorConfig,
    count_trainable_parameters,
    gaussian_patch_nll,
)
from bird_song.vocoder import VocoderMelScaler, VocoderSpectrogramConfig
from bird_song.vocoder_data import BigVGANMelDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    raw_model: ConditionalSpectrogramTransformer,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = total_items = 0
    for batch_index, (mels, labels, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        mels = mels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mean, log_scale = model(mels, labels)
        loss = gaussian_patch_nll(mels, mean, log_scale, raw_model)
        total_loss += float(loss) * labels.numel()
        total_items += labels.numel()
    if total_items == 0:
        raise RuntimeError("validation loader produced no batches")
    return total_loss / total_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a species-conditional autoregressive Transformer on BigVGAN log-mels."
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/bigvgan_mels/mel_manifest.csv",
    )
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/bigvgan_mels")
    parser.add_argument(
        "--vocoder-config",
        type=Path,
        default=PROJECT_ROOT / "configs/bigvgan_spectrogram.json",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs/transformer_16x16.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/transformer_16x16_seed42",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive; workers cannot be negative")
    if args.min_delta < 0 or args.gradient_clip <= 0:
        raise ValueError("min-delta cannot be negative and gradient-clip must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    if args.eval_batches is not None and args.eval_batches < 1:
        raise ValueError("eval-batches must be positive")

    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    cache_manifest = pd.read_csv(args.cache_manifest)
    required = {"split", "name", "relative_mel_path"}
    missing = required - set(cache_manifest.columns)
    if missing:
        raise ValueError(f"cache manifest is missing columns: {sorted(missing)}")
    classes = tuple(DEFAULT_CLASSES)
    if set(cache_manifest["name"].astype(str).unique()) != set(classes):
        raise ValueError("cache classes do not match the three expected bird species")

    vocoder_config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    model_config = TransformerGeneratorConfig.from_json(args.model_config)
    expected_shape = (vocoder_config.n_mels, vocoder_config.expected_frames)
    model_shape = (model_config.height, model_config.width)
    if model_shape != expected_shape:
        raise ValueError(f"Transformer shape {model_shape} does not match BigVGAN mel shape {expected_shape}")
    if model_config.num_classes != len(classes):
        raise ValueError("Transformer class count does not match the three expected species")

    scaler = VocoderMelScaler.from_json(args.cache_root / "scaler.json")
    train_set = BigVGANMelDataset(
        args.cache_manifest,
        args.cache_root,
        "train",
        classes,
        vocoder_config,
        scaler,
        specaugment=args.specaugment,
    )
    validation_set = BigVGANMelDataset(
        args.cache_manifest,
        args.cache_root,
        "validation",
        classes,
        vocoder_config,
        scaler,
    )
    train_loader = make_cached_loader(train_set, args.batch_size, args.workers, training=True, seed=args.seed)
    validation_loader = make_cached_loader(validation_set, args.batch_size, args.workers)

    raw_model = ConditionalSpectrogramTransformer(model_config).to(device)
    trainable_parameters = count_trainable_parameters(raw_model)
    model: nn.Module = raw_model
    if args.compile_model:
        model = torch.compile(model)

    existing = [args.output_dir / name for name in ("config.json", "history.csv", "best.pt") if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {[path.name for path in existing]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "classes": classes,
            "model": model_config.to_dict(),
            "vocoder": vocoder_config.to_dict(),
            "scaler": scaler.to_dict(),
            "representation": "bigvgan_raw_logmel_scaled",
            "trainable_parameters": trainable_parameters,
            "device_used": str(device),
        },
    )

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_validation_nll = float("inf")
    stale_epochs = 0
    history_path = args.output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        fields = ["epoch", "train_nll", "validation_nll", "lr", "seconds"]
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            total_train_loss = total_train_items = 0
            for batch_index, (mels, labels, _) in enumerate(train_loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                mels = mels.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    mean, log_scale = model(mels, labels)
                    loss = gaussian_patch_nll(mels, mean, log_scale, raw_model)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(raw_model.parameters(), args.gradient_clip)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                total_train_loss += float(loss.detach()) * labels.numel()
                total_train_items += labels.numel()
            if total_train_items == 0:
                raise RuntimeError("training loader produced no batches")

            validation_nll = evaluate(model, raw_model, validation_loader, device, args.eval_batches)
            row = {
                "epoch": epoch,
                "train_nll": total_train_loss / total_train_items,
                "validation_nll": validation_nll,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            history_file.flush()
            print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))

            if validation_nll < best_validation_nll - args.min_delta:
                best_validation_nll = validation_nll
                stale_epochs = 0
                atomic_torch_save(
                    {
                        "format_version": 2,
                        "model_type": "conditional_autoregressive_bigvgan_mel_transformer",
                        "model_state": cpu_state_dict(raw_model),
                        "model_config": model_config.to_dict(),
                        "classes": list(classes),
                        "vocoder_config": vocoder_config.to_dict(),
                        "scaler": scaler.to_dict(),
                        "representation": "bigvgan_raw_logmel_scaled",
                        "trainable_parameters": trainable_parameters,
                        "seed": args.seed,
                        "epoch": epoch,
                        "metrics": row | {"best_validation_nll": best_validation_nll},
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
