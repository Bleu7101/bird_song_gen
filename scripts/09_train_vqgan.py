from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from bird_song.config import SpectrogramConfig
from bird_song.generation.evaluation import detail_metrics
from bird_song.generation.vqgan import ConditionalVQGAN, PatchDiscriminator, VQGANConfig, count_parameters, reconstruction_loss
from bird_song.runtime import atomic_torch_save, choose_device, save_json, seed_everything
from bird_song.transformer.data import CachedSpectrogramDataset, make_cached_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VQGAN-style sharp spectrogram tokenizer and decoder.")
    parser.add_argument("--cache-manifest", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms/spectrogram_manifest.csv")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/vqgan.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/vqgan")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--discriminator-learning-rate", type=float, default=2e-4)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch size must be positive; workers cannot be negative")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    manifest = pd.read_csv(args.cache_manifest)
    classes = tuple(sorted(manifest["name"].unique()))
    spectrogram_config = SpectrogramConfig.from_json(args.spectrogram_config)
    config = VQGANConfig.from_dict(json.loads(args.model_config.read_text(encoding="utf-8")))
    if len(classes) != config.num_classes or (spectrogram_config.n_mels, spectrogram_config.spectrogram_width) != (128, 128):
        raise ValueError("VQGAN config must match the existing three-class 128 x 128 representation")
    train_set = CachedSpectrogramDataset(args.cache_manifest, args.cache_root, "train", classes, 128, specaugment=False)
    validation_set = CachedSpectrogramDataset(args.cache_manifest, args.cache_root, "validation", classes, 128, specaugment=False)
    train_loader = make_cached_loader(train_set, args.batch_size, args.workers, training=True, seed=args.seed)
    validation_loader = make_cached_loader(validation_set, args.batch_size, args.workers)
    model = ConditionalVQGAN(config).to(device)
    discriminator = PatchDiscriminator(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_learning_rate, betas=(0.5, 0.9))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in ("config.json", "history.csv", "best.pt") if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {[path.name for path in existing]}; pass --overwrite")
    save_json(args.output_dir / "config.json", {
        **vars(args), "classes": classes, "model": config.to_dict(), "spectrogram": spectrogram_config.to_dict(),
        "model_parameters": count_parameters(model), "discriminator_parameters": count_parameters(discriminator), "device_used": str(device),
    })
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history_path = args.output_dir / "history.csv"
    best_detail = float("-inf")
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        fields = ["epoch", "reconstruction_loss", "vq_loss", "generator_adversarial_loss", "discriminator_loss", "time_detail_ratio", "frequency_detail_ratio", "seconds"]
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            discriminator.train()
            totals = {field: [] for field in fields[1:-1]}
            for batch_index, (real, labels, _) in enumerate(train_loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                real = real.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    reconstruction, _, _, vq_loss = model(real, labels)
                    rec_loss = reconstruction_loss(real, reconstruction)
                    generator_adv = -discriminator(reconstruction).mean()
                    generator_total = rec_loss + vq_loss + args.adversarial_weight * generator_adv
                scaler.scale(generator_total).backward()
                scaler.step(optimizer)
                scaler.update()

                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    real_score = discriminator(real)
                    fake_score = discriminator(reconstruction.detach())
                    discriminator_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
                scaler.scale(discriminator_loss).backward()
                scaler.step(discriminator_optimizer)
                scaler.update()
                totals["reconstruction_loss"].append(float(rec_loss.detach()))
                totals["vq_loss"].append(float(vq_loss.detach()))
                totals["generator_adversarial_loss"].append(float(generator_adv.detach()))
                totals["discriminator_loss"].append(float(discriminator_loss.detach()))

            model.eval()
            real_batches: list[torch.Tensor] = []
            reconstruction_batches: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch_index, (real, labels, _) in enumerate(validation_loader):
                    if args.max_batches is not None and batch_index >= args.max_batches:
                        break
                    real = real.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    reconstruction_batches.append(model(real, labels)[0].cpu())
                    real_batches.append(real.cpu())
            validation_detail = detail_metrics(torch.cat(real_batches), torch.cat(reconstruction_batches))
            row = {field: sum(values) / max(len(values), 1) for field, values in totals.items()}
            row.update({"epoch": epoch, "time_detail_ratio": validation_detail["time_detail_ratio"], "frequency_detail_ratio": validation_detail["frequency_detail_ratio"], "seconds": time.perf_counter() - started})
            writer.writerow(row)
            history_file.flush()
            print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))
            checkpoint = {
                "format_version": 1, "model_type": "conditional_vqgan_spectrogram_tokenizer", "model_state": model.state_dict(), "discriminator_state": discriminator.state_dict(),
                "optimizer": optimizer.state_dict(), "discriminator_optimizer": discriminator_optimizer.state_dict(), "model_config": config.to_dict(), "classes": list(classes), "spectrogram_config": spectrogram_config.to_dict(), "epoch": epoch, "seed": args.seed, "metrics": row,
            }
            atomic_torch_save(checkpoint, args.output_dir / "last.pt")
            combined_detail = (row["time_detail_ratio"] + row["frequency_detail_ratio"]) / 2
            if combined_detail > best_detail:
                best_detail = combined_detail
                atomic_torch_save(checkpoint, args.output_dir / "best.pt")
    print(f"Best reconstruction detail score: {best_detail:.4f}")
    print(f"Checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
