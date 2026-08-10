from __future__ import annotations

import argparse
import json
from pathlib import Path

from bird_song.generation.checkpoint_evaluation import (
    EVALUATION_SEEDS,
    audit_pools,
    build_report_charts,
    evaluate,
    write_checksums,
)
from bird_song.generation.checkpoint_pool import verify_checkpoint
from bird_song.runtime import choose_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAE_CHECKPOINT = PROJECT_ROOT / "artifacts/models/vae/conditional_vae_v3/conditional_vae_v3_best.pt"
DEFAULT_VAE_BANK = PROJECT_ROOT / "artifacts/models/vae/conditional_vae_v3/class_conditional_posterior_bank.pt"
DEFAULT_DIFFUSION_CHECKPOINT = Path(r"C:\Users\Harvey\Desktop\conditional_diffusion\conditional_diffusion_best_low_batch_100_eps_weighted.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit, evaluate, and package frozen VAE-v3/diffusion generator pools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "evaluate", "package"):
        command = subparsers.add_parser(name)
        command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
        command.add_argument("--pool-root", type=Path, default=PROJECT_ROOT / "runs/generator_checkpoint_evaluation/pools")
        command.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/generator_checkpoint_evaluation_2026-08-10")
        command.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
        command.add_argument("--diffusion-checkpoint", type=Path, default=DEFAULT_DIFFUSION_CHECKPOINT)
        command.add_argument("--train-manifest", type=Path, default=PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_train.csv")
        command.add_argument("--validation-manifest", type=Path, default=PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_validation.csv")
        command.add_argument("--test-manifest", type=Path, default=PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_test.csv")
        command.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
        command.add_argument("--seeds", nargs="+", type=int, default=list(EVALUATION_SEEDS))
        command.add_argument("--device", default="auto")
        command.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    pool_root = args.pool_root.resolve()
    if args.command == "audit":
        verify_checkpoint(args.vae_checkpoint.resolve(), "vae_v3")
        verify_checkpoint(args.diffusion_checkpoint.resolve(), "diffusion")
        result = audit_pools(project_root, pool_root, seeds=args.seeds)
        print(json.dumps(result, indent=2))
        return

    vae_checkpoint = args.vae_checkpoint.resolve()
    diffusion_checkpoint = args.diffusion_checkpoint.resolve()
    # Hash verification is intentionally quiet on success.  A mismatch raises
    # one concise error before any generated-sample conclusions are written.
    verify_checkpoint(vae_checkpoint, "vae_v3")
    verify_checkpoint(diffusion_checkpoint, "diffusion")
    device = choose_device(args.device)
    summary = evaluate(
        project_root=project_root,
        pool_root=pool_root,
        report_dir=args.report_dir,
        crnn_checkpoint=project_root / "artifacts/models/classifier/selected_crnn/best.pt",
        residual_checkpoint=project_root / "artifacts/models/classifier/Harvey_classifier/best.pt",
        vae_checkpoint=vae_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        test_manifest=args.test_manifest.resolve(),
        validation_manifest=args.validation_manifest.resolve(),
        train_manifest=args.train_manifest.resolve(),
        cache_root=args.cache_root.resolve(),
        device=device,
        batch_size=args.batch_size,
        seeds=args.seeds,
    )
    if args.command == "package":
        build_report_charts(args.report_dir.resolve())
        write_checksums(args.report_dir.resolve())
        print(json.dumps(summary, indent=2))
        print(f"report={args.report_dir.resolve()}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
