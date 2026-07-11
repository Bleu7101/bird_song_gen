from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from bird_song.config import DEFAULT_CLASSES, SpectrogramConfig
from bird_song.data import ManifestDataset, make_loader, resolve_dataset_root
from bird_song.classifier.model import BirdSongCNN
from bird_song.runtime import choose_device, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short, non-training throughput benchmark for the classifier.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/benchmarks/model_throughput.csv")
    parser.add_argument("--data-pipeline", action="store_true", help="Also time test-manifest audio decoding and log-mel creation.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_test.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-batches", type=int, default=20)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if args.checkpoint:
        model, classes, config, _ = load_checkpoint(args.checkpoint, device)
    else:
        classes, config = DEFAULT_CLASSES, SpectrogramConfig.from_json(args.spectrogram_config)
        model = BirdSongCNN(len(classes)).to(device)
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            inputs = torch.randn(batch_size, 1, config.n_mels, config.spectrogram_width, device=device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(args.warmup):
                model(inputs)
            synchronize(device)
            started = time.perf_counter()
            for _ in range(args.iterations):
                model(inputs)
            synchronize(device)
            seconds = time.perf_counter() - started
            row = {
                "device": str(device),
                "batch_size": batch_size,
                "iterations": args.iterations,
                "samples_per_second": batch_size * args.iterations / seconds,
                "milliseconds_per_batch": seconds * 1000 / args.iterations,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
            }
            rows.append(row)
            print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.data_pipeline:
        dataset = ManifestDataset(args.manifest, resolve_dataset_root(PROJECT_ROOT, args.dataset_root), classes, config)
        loader = make_loader(dataset, max(args.batch_sizes), args.workers)
        seen = 0
        started = time.perf_counter()
        for index, (specs, _, _) in enumerate(loader):
            seen += len(specs)
            if index + 1 >= args.data_batches:
                break
        seconds = time.perf_counter() - started
        print(f"data_pipeline_samples_per_second={seen / seconds:.2f} samples={seen} seconds={seconds:.3f}")
    print(f"Benchmark CSV: {args.output}")


if __name__ == "__main__":
    main()
