from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bird_song.evaluation.generated_to_test_mse import evaluate_generated_to_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated-to-test and real-to-real nearest-neighbour MSE.")
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-metadata", type=Path, default=None, help="Optional pool generation.json metadata file.")
    parser.add_argument(
        "--comparison-domain",
        choices=("classifier_input", "standardized_logmel"),
        default="classifier_input",
        help="Numeric domain in which both arrays are compared.",
    )
    parser.add_argument(
        "--generated-input-domain",
        choices=("classifier_input", "standardized_logmel"),
        default="classifier_input",
        help="Numeric domain of generated arrays before comparison.",
    )
    parser.add_argument(
        "--generated-normalization-stats",
        type=Path,
        default=None,
        help="Train-only normalization_stats.json for standardized generated arrays converted to classifier input.",
    )
    parser.add_argument(
        "--test-input-domain",
        choices=("classifier_input", "standardized_logmel"),
        default="classifier_input",
        help="Numeric domain of test arrays before comparison.",
    )
    parser.add_argument(
        "--test-normalization-stats",
        type=Path,
        default=None,
        help="Train-only normalization_stats.json required for standardized_logmel test arrays.",
    )
    args = parser.parse_args()
    result = evaluate_generated_to_test(
        generated_manifest=args.generated_manifest,
        generated_root=args.generated_root,
        test_manifest=args.test_manifest,
        test_root=args.test_root,
        generated_metadata=args.generated_metadata,
        comparison_domain=args.comparison_domain,
        generated_input_domain=args.generated_input_domain,
        generated_normalization_stats=args.generated_normalization_stats,
        test_input_domain=args.test_input_domain,
        test_normalization_stats=args.test_normalization_stats,
    )
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    result.write(args.output, git_revision=revision)
    print(f"wrote={args.output.resolve()}")


if __name__ == "__main__":
    main()
