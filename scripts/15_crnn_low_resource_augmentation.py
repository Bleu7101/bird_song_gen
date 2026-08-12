from __future__ import annotations

import argparse
import json
from pathlib import Path

from bird_song.augmentation.low_resource import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_POOL_SEEDS,
    DEFAULT_RATIOS,
    DEFAULT_REAL_PER_SPECIES,
    DEFAULT_REAL_SUBSET_SEEDS,
    DEFAULT_STEPS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALIDATE_EVERY,
    audit_inputs,
    run_sweep,
    select_and_evaluate,
)
from bird_song.augmentation.low_resource_report import package_report
from bird_song.config import SpectrogramConfig
from bird_song.data import resolve_spectrogram_cache_root
from bird_song.runtime import choose_device, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs/crnn_low_resource_augmentation/v3"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/crnn_low_resource_augmentation_2026-08-12"
DEFAULT_GENERATOR_EVALUATION_PROTOCOL = (
    PROJECT_ROOT / "reports/generator_checkpoint_evaluation_2026-08-12/protocol.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the matched 50-real-per-species CRNN augmentation experiment with "
            "existing VAE-v3 and diffusion pools."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "train-sweep", "select-evaluate", "package", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
        command.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
        command.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
        command.add_argument(
            "--train-manifest",
            type=Path,
            default=PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_train.csv",
        )
        command.add_argument(
            "--validation-manifest",
            type=Path,
            default=(
                PROJECT_ROOT
                / "manifests/content_safe_v2/full_dataset_validation_generator_safe.csv"
            ),
        )
        command.add_argument(
            "--test-manifest",
            type=Path,
            default=PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_test.csv",
        )
        command.add_argument(
            "--real-cache",
            type=Path,
            default=PROJECT_ROOT / "artifacts/spectrograms",
        )
        command.add_argument(
            "--pool-root",
            type=Path,
            default=PROJECT_ROOT / "runs/generator_checkpoint_evaluation/pools",
        )
        command.add_argument(
            "--spectrogram-config",
            type=Path,
            default=PROJECT_ROOT / "configs/spectrogram.json",
        )
        command.add_argument("--real-per-species", type=int, default=DEFAULT_REAL_PER_SPECIES)
        command.add_argument("--ratios", nargs="+", type=int, default=list(DEFAULT_RATIOS))
        command.add_argument(
            "--subset-seeds", nargs="+", type=int, default=list(DEFAULT_REAL_SUBSET_SEEDS)
        )
        command.add_argument("--train-seeds", nargs="+", type=int, default=list(DEFAULT_TRAIN_SEEDS))
        command.add_argument("--pool-seeds", nargs="+", type=int, default=list(DEFAULT_POOL_SEEDS))
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        command.add_argument("--device", default="cuda")
    for name in ("train-sweep", "run"):
        command = subparsers.choices[name]
        command.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        command.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
        command.add_argument("--overwrite", action="store_true")
    for name in ("select-evaluate", "package", "run"):
        command = subparsers.choices[name]
        command.add_argument(
            "--evaluation-output",
            type=Path,
            default=None,
            help="Evaluation JSON (default: <run-root>/evaluation.json).",
        )
        command.add_argument("--overwrite-report", action="store_true")
    for name in ("package", "run"):
        subparsers.choices[name].add_argument(
            "--generator-evaluation-protocol",
            type=Path,
            default=DEFAULT_GENERATOR_EVALUATION_PROTOCOL,
            help="Schema-3 generator-evaluation protocol that records pool refresh actions.",
        )
    return parser


def _resolved_inputs(args: argparse.Namespace) -> dict[str, Path]:
    project_root = args.project_root.resolve()
    return {
        "project_root": project_root,
        "run_root": args.run_root.resolve(),
        "report_dir": args.report_dir.resolve(),
        "source_train_manifest": args.train_manifest.resolve(),
        "validation_manifest": args.validation_manifest.resolve(),
        "test_manifest": args.test_manifest.resolve(),
        "cache_root": resolve_spectrogram_cache_root(project_root, args.real_cache),
        "pool_root": args.pool_root.resolve(),
        "spectrogram_config_path": args.spectrogram_config.resolve(),
        "generator_evaluation_protocol": getattr(
            args,
            "generator_evaluation_protocol",
            DEFAULT_GENERATOR_EVALUATION_PROTOCOL,
        ).resolve(),
    }


def _audit(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    return audit_inputs(
        project_root=paths["project_root"],
        run_root=paths["run_root"],
        source_train_manifest=paths["source_train_manifest"],
        validation_manifest=paths["validation_manifest"],
        test_manifest=paths["test_manifest"],
        cache_root=paths["cache_root"],
        pool_root=paths["pool_root"],
        spectrogram_config_path=paths["spectrogram_config_path"],
        real_per_species=args.real_per_species,
        ratios=args.ratios,
        real_subset_seeds=args.subset_seeds,
        train_seeds=args.train_seeds,
        pool_seeds=args.pool_seeds,
    )


def main() -> None:
    args = build_parser().parse_args()
    paths = _resolved_inputs(args)
    evaluation_path = (
        args.evaluation_output.resolve()
        if getattr(args, "evaluation_output", None)
        else paths["run_root"] / "evaluation.json"
    )
    if args.command == "audit":
        print(json.dumps(_audit(args, paths), indent=2))
        return

    config = SpectrogramConfig.from_json(paths["spectrogram_config_path"])
    if args.command in ("train-sweep", "run"):
        audit = _audit(args, paths)
        save_json(paths["run_root"] / "input_audit.json", audit)
        print(
            f"audit_ok training_runs={audit['expected_training_runs']} "
            f"generated_arrays={audit['validated_generated_arrays']}",
            flush=True,
        )
        device = choose_device(args.device)
        run_sweep(
            project_root=paths["project_root"],
            run_root=paths["run_root"],
            source_train_manifest=paths["source_train_manifest"],
            validation_manifest=paths["validation_manifest"],
            cache_root=paths["cache_root"],
            pool_root=paths["pool_root"],
            spectrogram_config_path=paths["spectrogram_config_path"],
            config=config,
            device=device,
            real_per_species=args.real_per_species,
            ratios=args.ratios,
            real_subset_seeds=args.subset_seeds,
            train_seeds=args.train_seeds,
            pool_seeds=args.pool_seeds,
            steps=args.steps,
            validate_every=args.validate_every,
            batch_size=args.batch_size,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        if args.command == "train-sweep":
            return

    if args.command in ("select-evaluate", "run"):
        device = choose_device(args.device)
        select_and_evaluate(
            project_root=paths["project_root"],
            run_root=paths["run_root"],
            test_manifest=paths["test_manifest"],
            cache_root=paths["cache_root"],
            config=config,
            device=device,
            output_path=evaluation_path,
            real_per_species=args.real_per_species,
            ratios=args.ratios,
            real_subset_seeds=args.subset_seeds,
            train_seeds=args.train_seeds,
            pool_seeds=args.pool_seeds,
            batch_size=args.batch_size,
            workers=args.workers,
            overwrite=args.overwrite_report,
        )
        if args.command == "select-evaluate":
            print(f"evaluation={evaluation_path}")
            return

    if args.command in ("package", "run"):
        report = package_report(
            project_root=paths["project_root"],
            run_root=paths["run_root"],
            evaluation_path=evaluation_path,
            report_dir=paths["report_dir"],
            source_train_manifest=paths["source_train_manifest"],
            validation_manifest=paths["validation_manifest"],
            test_manifest=paths["test_manifest"],
            cache_root=paths["cache_root"],
            pool_root=paths["pool_root"],
            spectrogram_config_path=paths["spectrogram_config_path"],
            generator_evaluation_protocol=paths["generator_evaluation_protocol"],
            overwrite=args.overwrite_report,
        )
        print(f"report={report}")


if __name__ == "__main__":
    main()
