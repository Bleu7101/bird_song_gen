from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import torch

from bird_song.config import SpectrogramConfig
from bird_song.generation.token_transformer import ConditionalTokenTransformer, TokenTransformerConfig, count_parameters
from bird_song.generation.vqgan import ConditionalVQGAN, VQGANConfig
from bird_song.runtime import atomic_torch_save, choose_device, save_json, seed_everything
from bird_song.transformer.data import CachedSpectrogramDataset, make_cached_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a token transformer over a VQGAN spectrogram codebook.")
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument("--token-config", type=Path, default=PROJECT_ROOT / "configs/token_transformer.json")
    parser.add_argument("--cache-manifest", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms/spectrogram_manifest.csv")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/token_transformer")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model: ConditionalTokenTransformer, vqgan: ConditionalVQGAN, loader, device: torch.device, max_batches: int | None) -> float:
    model.eval()
    vqgan.eval()
    total = count = 0.0
    for batch_index, (real, labels, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices = vqgan.encode(real)[1]
        loss = model.loss(indices, labels)
        total += float(loss) * labels.numel()
        count += labels.numel()
    return total / max(count, 1.0)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch size must be positive; workers cannot be negative")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    vq_checkpoint = torch.load(args.vqgan_checkpoint, map_location=device, weights_only=True)
    vq_config = VQGANConfig.from_dict(vq_checkpoint["model_config"])
    token_config = TokenTransformerConfig.from_dict(json.loads(args.token_config.read_text(encoding="utf-8")))
    if token_config.codebook_size != vq_config.codebook_size or token_config.token_grid != vq_config.latent_grid:
        raise ValueError("token transformer config must match the VQGAN codebook and latent grid")
    vqgan = ConditionalVQGAN(vq_config).to(device)
    vqgan.load_state_dict(vq_checkpoint["model_state"])
    vqgan.eval()
    for parameter in vqgan.parameters():
        parameter.requires_grad_(False)
    manifest = pd.read_csv(args.cache_manifest)
    classes = tuple(sorted(manifest["name"].unique()))
    spectrogram_config = SpectrogramConfig.from_json(args.spectrogram_config)
    if len(classes) != token_config.num_classes or (spectrogram_config.n_mels, spectrogram_config.spectrogram_width) != (128, 128):
        raise ValueError("token transformer must match the existing three-class 128 x 128 representation")
    train_set = CachedSpectrogramDataset(args.cache_manifest, args.cache_root, "train", classes, 128, specaugment=False)
    validation_set = CachedSpectrogramDataset(args.cache_manifest, args.cache_root, "validation", classes, 128, specaugment=False)
    train_loader = make_cached_loader(train_set, args.batch_size, args.workers, training=True, seed=args.seed)
    validation_loader = make_cached_loader(validation_set, args.batch_size, args.workers)
    model = ConditionalTokenTransformer(token_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in ("config.json", "history.csv", "best.pt") if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {[path.name for path in existing]}; pass --overwrite")
    save_json(args.output_dir / "config.json", {
        **vars(args), "classes": classes, "token_model": token_config.to_dict(), "vqgan_model": vq_config.to_dict(), "spectrogram": spectrogram_config.to_dict(), "trainable_parameters": count_parameters(model), "device_used": str(device),
    })
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_validation = float("inf")
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "train_nll", "validation_nll", "lr", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            losses: list[float] = []
            for batch_index, (real, labels, _) in enumerate(train_loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                real = real.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # ``no_grad`` keeps ordinary tensors that autograd can consume
                # later; inference tensors cannot be passed into the trainable
                # transformer loss on some PyTorch versions.
                with torch.no_grad():
                    indices = vqgan.encode(real)[1]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = model.loss(indices, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            validation_nll = evaluate(model, vqgan, validation_loader, device, args.max_batches)
            row = {"epoch": epoch, "train_nll": sum(losses) / max(len(losses), 1), "validation_nll": validation_nll, "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}
            writer.writerow(row)
            history_file.flush()
            print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))
            checkpoint = {
                "format_version": 1, "model_type": "conditional_vqgan_token_transformer", "model_state": model.state_dict(), "model_config": token_config.to_dict(), "vqgan_config": vq_config.to_dict(), "classes": list(classes), "spectrogram_config": spectrogram_config.to_dict(), "vqgan_checkpoint": str(args.vqgan_checkpoint), "epoch": epoch, "seed": args.seed, "validation_nll": validation_nll,
            }
            if validation_nll < best_validation:
                best_validation = validation_nll
                atomic_torch_save(checkpoint, args.output_dir / "best.pt")
            scheduler.step()
    print(f"Best validation NLL: {best_validation:.4f}")
    print(f"Checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
