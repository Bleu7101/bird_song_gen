from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from bird_song.config import SpectrogramConfig
from bird_song.data import ManifestDataset, make_loader, resolve_dataset_root
from bird_song.classifier.model import BirdSongCNN
from bird_song.runtime import atomic_torch_save, choose_device, save_json, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bird-song species classifier.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--train-manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_train.csv")
    parser.add_argument("--val-manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_validation.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run in --output-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Load one batch and run one forward pass, without training.")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total_items = 0
    for specs, labels, _ in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(specs)
        total_loss += float(criterion(logits, labels)) * labels.numel()
        total_correct += int((logits.argmax(1) == labels).sum())
        total_items += labels.numel()
    return total_loss / total_items, total_correct / total_items


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    classes = tuple(sorted(pd.read_csv(args.train_manifest)["name"].unique()))
    config = SpectrogramConfig.from_json(args.spectrogram_config)
    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    train_set = ManifestDataset(args.train_manifest, dataset_root, classes, config, training=True)
    val_set = ManifestDataset(args.val_manifest, dataset_root, classes, config, training=False)
    train_loader = make_loader(train_set, args.batch_size, args.workers, training=True, balanced=args.balanced_sampler)
    val_loader = make_loader(val_set, args.batch_size, args.workers)

    model = BirdSongCNN(num_classes=len(classes)).to(device)
    if args.dry_run:
        specs, labels, paths = next(iter(train_loader))
        with torch.inference_mode():
            logits = model(specs.to(device))
        print(f"device={device} batch={tuple(specs.shape)} logits={tuple(logits.shape)}")
        print(f"classes={classes}")
        print(f"first_file={paths[0]}")
        print("Dry run complete; no optimizer step was performed.")
        return

    existing_outputs = [args.output_dir / name for name in ("config.json", "history.csv", "best.pt")]
    existing_outputs = [path for path in existing_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(
            f"Refusing to overwrite existing run files in {args.output_dir}: {names}. "
            "Choose a new --output-dir or pass --overwrite explicitly."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {**vars(args), "dataset_root": str(dataset_root), "classes": classes, "spectrogram": config.to_dict()},
    )
    raw_model = model
    if args.compile_model:
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    log_path = args.output_dir / "history.csv"
    best_accuracy = float("-inf")
    epochs_without_improvement = 0

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "lr", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            running_loss = item_count = 0
            for specs, labels, _ in train_loader:
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = criterion(model(specs), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach()) * labels.numel()
                item_count += labels.numel()

            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": running_loss / item_count,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            log_file.flush()
            print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                epochs_without_improvement = 0
                atomic_torch_save(
                    {
                        "format_version": 1,
                        "model_state": raw_model.state_dict(),
                        "model_config": raw_model.metadata(),
                        "classes": list(classes),
                        "spectrogram_config": config.to_dict(),
                        "epoch": epoch,
                        "best_validation_accuracy": best_accuracy,
                    },
                    args.output_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1
            scheduler.step()
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
