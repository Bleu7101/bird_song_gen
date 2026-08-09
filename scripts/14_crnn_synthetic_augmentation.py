from __future__ import annotations

import argparse
import json
from pathlib import Path

from bird_song.augmentation.experiment import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RATIOS,
    DEFAULT_SEEDS,
    DEFAULT_STEPS,
    DEFAULT_VALIDATE_EVERY,
    audit_inputs,
    run_sweep,
    select_and_evaluate,
)
from bird_song.config import SpectrogramConfig
from bird_song.data import resolve_spectrogram_cache_root
from bird_song.runtime import choose_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache-backed VAE-v3/diffusion CRNN augmentation evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "train-sweep", "select-evaluate", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--real-cache", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
        command.add_argument(
            "--train-manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_train.csv"
        )
        command.add_argument(
            "--validation-manifest",
            type=Path,
            default=PROJECT_ROOT / "manifests/full_dataset_validation.csv",
        )
        command.add_argument(
            "--test-manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_test.csv"
        )
        command.add_argument(
            "--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json"
        )
        command.add_argument(
            "--baseline-metrics",
            type=Path,
            default=PROJECT_ROOT / "classifier_artifacts/selected_crnn/metrics.json",
        )
        command.add_argument(
            "--generated-cache",
            type=Path,
            default=PROJECT_ROOT / "artifacts/generated_spectrograms",
        )
        command.add_argument(
            "--run-root",
            type=Path,
            default=PROJECT_ROOT
            / "runs/crnn_synthetic_augmentation"
            / ("cache_sweep" if name == "select-evaluate" else "cache_sweep_v2"),
        )
        command.add_argument("--ratios", nargs="+", type=int, default=list(DEFAULT_RATIOS))
        command.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
        command.add_argument(
            "--generators",
            nargs="+",
            choices=("vae_v3", "diffusion"),
            default=["vae_v3", "diffusion"],
        )
        command.add_argument("--device", default="cuda")
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    for name in ("train-sweep", "run"):
        command = subparsers.choices[name]
        command.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        command.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
        command.add_argument("--overwrite", action="store_true")
    for name in ("select-evaluate", "run"):
        command = subparsers.choices[name]
        command.add_argument(
            "--report-output",
            type=Path,
            default=None,
            help="Output JSON (default: <run-root>/selection_evaluation_recomputed.json).",
        )
        command.add_argument("--overwrite-report", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    real_cache = resolve_spectrogram_cache_root(PROJECT_ROOT, args.real_cache)
    generated_cache = args.generated_cache.resolve()
    spectrogram_config_path = args.spectrogram_config.resolve()
    config = SpectrogramConfig.from_json(spectrogram_config_path)
    train_manifest = args.train_manifest.resolve()
    validation_manifest = args.validation_manifest.resolve()
    test_manifest = args.test_manifest.resolve()
    if args.command == "audit":
        print(
            json.dumps(
                audit_inputs(
                    train_manifest,
                    real_cache,
                    generated_cache,
                    config,
                    args.ratios,
                    args.generators,
                ),
                indent=2,
            )
        )
        return
    report_output = None
    if args.command in ("select-evaluate", "run"):
        report_output = (args.report_output or args.run_root / "selection_evaluation_recomputed.json").resolve()
        if report_output.exists() and not args.overwrite_report:
            raise FileExistsError(
                f"Evaluation report already exists: {report_output}; pass --overwrite-report intentionally"
            )
    device = choose_device(args.device)
    if args.command in ("train-sweep", "run"):
        run_sweep(
            project_root=PROJECT_ROOT,
            generators=args.generators,
            ratios=args.ratios,
            seeds=args.seeds,
            run_root=args.run_root,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            real_cache_root=real_cache,
            generated_cache_root=generated_cache,
            spectrogram_config_path=spectrogram_config_path,
            config=config,
            device=device,
            steps=args.steps,
            validate_every=args.validate_every,
            batch_size=args.batch_size,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        if args.command == "train-sweep":
            return
    assert report_output is not None
    report = select_and_evaluate(
        project_root=PROJECT_ROOT,
        generators=args.generators,
        ratios=args.ratios,
        seeds=args.seeds,
        run_root=args.run_root,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        real_cache_root=real_cache,
        generated_cache_root=generated_cache,
        baseline_metrics=args.baseline_metrics.resolve(),
        spectrogram_config_path=spectrogram_config_path,
        output_path=report_output,
        config=config,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        overwrite_report=args.overwrite_report,
    )
    print(f"report={report}")


if __name__ == "__main__":
    main()
