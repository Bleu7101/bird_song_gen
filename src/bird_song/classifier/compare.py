from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bird_song.classifier.model import ARCHITECTURES


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classifier architectures under one controlled protocol and summarize validation results."
    )
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 777],
        help="Repeat every architecture with these random seeds (three are recommended for reporting).",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--spectrogram-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts/spectrograms",
        help="Historical real-audio spectrogram cache root (default: artifacts/spectrograms).",
    )
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--train-manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_train.csv")
    parser.add_argument("--val-manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_validation.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/classifier_architectures")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse runs that already contain config.json, history.csv, and best.pt.",
    )
    return parser.parse_args()


def run_is_complete(run_dir: Path) -> bool:
    return all((run_dir / filename).is_file() for filename in ("config.json", "history.csv", "best.pt"))


def validate_existing_run(
    args: argparse.Namespace,
    run_dir: Path,
    architecture: str,
    seed: int,
) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    expected = {
        "architecture": architecture,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "width": args.width,
        "dropout": args.dropout,
        "balanced_sampler": args.balanced_sampler,
        "compile_model": args.compile_model,
        "spectrogram_config": str(args.spectrogram_config),
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "dataset_root": str(args.dataset_root),
        "spectrogram_cache": str(args.spectrogram_cache),
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        details = ", ".join(f"{key}: saved={saved!r}, requested={requested!r}" for key, (saved, requested) in mismatches.items())
        raise ValueError(f"Cannot reuse {run_dir}; its protocol differs ({details})")


def training_command(args: argparse.Namespace, architecture: str, seed: int, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bird_song.classifier.train",
        "--architecture",
        architecture,
        "--seed",
        str(seed),
        "--spectrogram-config",
        str(args.spectrogram_config),
        "--train-manifest",
        str(args.train_manifest),
        "--val-manifest",
        str(args.val_manifest),
        "--spectrogram-cache",
        str(args.spectrogram_cache),
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--label-smoothing",
        str(args.label_smoothing),
        "--patience",
        str(args.patience),
        "--width",
        str(args.width),
        "--dropout",
        str(args.dropout),
        "--device",
        args.device,
    ]
    command.extend(("--dataset-root", str(args.dataset_root)))
    if args.balanced_sampler:
        command.append("--balanced-sampler")
    if args.compile_model:
        command.append("--compile")
    if args.overwrite:
        command.append("--overwrite")
    return command


def read_run(run_dir: Path, architecture: str, seed: int) -> dict[str, float | int | str]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv")
    if history.empty:
        raise ValueError(f"Training history is empty: {run_dir / 'history.csv'}")
    selected = history.loc[history["val_accuracy"].idxmax()]
    return {
        "architecture": architecture,
        "seed": seed,
        "trainable_parameters": int(config["trainable_parameters"]),
        "best_epoch": int(selected["epoch"]),
        "validation_accuracy": float(selected["val_accuracy"]),
        "validation_macro_f1": float(selected["val_macro_f1"]),
        "validation_loss": float(selected["val_loss"]),
        "training_seconds": float(history["seconds"].sum()),
        "run_dir": str(run_dir.resolve()),
    }


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        runs.groupby("architecture", sort=False)
        .agg(
            runs=("seed", "count"),
            trainable_parameters=("trainable_parameters", "first"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_accuracy_std=("validation_accuracy", "std"),
            validation_macro_f1_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_std=("validation_macro_f1", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            training_seconds_total=("training_seconds", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values("validation_accuracy_mean", ascending=False)


def mean_and_sd(mean: float, standard_deviation: float, runs: int) -> str:
    return f"{mean:.2%} +/- {standard_deviation:.2%}" if runs > 1 else f"{mean:.2%} (SD n/a)"


def write_markdown(summary: pd.DataFrame, args: argparse.Namespace, output_path: Path) -> None:
    lines = [
        "# Classifier architecture comparison",
        "",
        "All architectures used the same recording-ID-isolated historical train/validation manifests, preprocessing, optimizer settings, "
        "early-stopping rule, and seed set. The held-out test split was not used for architecture selection.",
        "",
        f"Seeds: `{', '.join(map(str, args.seeds))}`. Maximum epochs: {args.epochs}. "
        f"Early-stopping patience: {args.patience}. Batch size: {args.batch_size}. "
        f"Learning rate: {args.learning_rate:g}. Weight decay: {args.weight_decay:g}. "
        f"Width: {args.width}. Dropout: {args.dropout:g}.",
        "",
        "Each run's selected epoch is the one with the highest validation accuracy. Values are mean +/- sample "
        "standard deviation when at least two seeds were run.",
        "",
        "| Architecture | Parameters | Runs | Validation accuracy | Validation macro F1 | Mean best epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.architecture} | {row.trainable_parameters:,} | {row.runs} | "
            f"{mean_and_sd(row.validation_accuracy_mean, row.validation_accuracy_std, row.runs)} | "
            f"{mean_and_sd(row.validation_macro_f1_mean, row.validation_macro_f1_std, row.runs)} | "
            f"{row.best_epoch_mean:.1f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(set(args.architectures)) != len(args.architectures):
        raise ValueError("--architectures contains duplicate names")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicate values")
    args.spectrogram_config = args.spectrogram_config.resolve()
    args.train_manifest = args.train_manifest.resolve()
    args.val_manifest = args.val_manifest.resolve()
    args.spectrogram_cache = args.spectrogram_cache.resolve()
    args.output_dir = args.output_dir.resolve()
    args.dataset_root = (args.dataset_root or PROJECT_ROOT / "bird_songs_dataset").resolve()

    rows = []
    for architecture in args.architectures:
        for seed in args.seeds:
            run_dir = args.output_dir / architecture / f"seed_{seed}"
            if args.skip_existing and run_is_complete(run_dir):
                validate_existing_run(args, run_dir, architecture, seed)
                print(f"Reusing complete run: {run_dir}")
            else:
                print(f"Training architecture={architecture} seed={seed}")
                subprocess.run(training_command(args, architecture, seed, run_dir), cwd=PROJECT_ROOT, check=True)
            rows.append(read_run(run_dir, architecture, seed))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    protocol = {
        key: value
        for key, value in vars(args).items()
        if key not in {"overwrite", "skip_existing"}
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps({**protocol, "project_root": str(PROJECT_ROOT)}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, args, args.output_dir / "comparison.md")
    print(summary.to_string(index=False))
    print(f"Comparison report: {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
