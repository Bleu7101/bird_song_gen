from __future__ import annotations

import argparse
import json
from pathlib import Path

from bird_song.generator_safe_validation import prepare_generator_safe_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the validation-only generator-safe manifest from the existing "
            "content_safe_v2 exact-duplicate ledger."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--source-validation",
        type=Path,
        default=PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_validation.csv",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests/content_safe_v2/full_dataset_test.csv",
    )
    parser.add_argument(
        "--historical-manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests/full_dataset_manifest.csv",
    )
    parser.add_argument(
        "--exact-duplicate-protocol",
        type=Path,
        default=PROJECT_ROOT / "manifests/content_safe_v2/protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "manifests/content_safe_v2/full_dataset_validation_generator_safe.csv"
        ),
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=None,
        help="Protocol JSON (default: sibling <output-stem>.protocol.json).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = prepare_generator_safe_validation(
        project_root=args.project_root.resolve(),
        source_validation_manifest=args.source_validation.resolve(),
        source_test_manifest=args.test_manifest.resolve(),
        historical_manifest=args.historical_manifest.resolve(),
        exact_duplicate_protocol=args.exact_duplicate_protocol.resolve(),
        output_validation_manifest=args.output.resolve(),
        output_protocol=args.protocol_output.resolve() if args.protocol_output else None,
    )
    print(json.dumps(protocol["validation_counts"], indent=2))
    print(f"validation_manifest={args.output.resolve()}")


if __name__ == "__main__":
    main()
