from __future__ import annotations

import argparse
from pathlib import Path

from bird_song.generation.speed_report import package_speed_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a matched generator-only speed comparison.")
    parser.add_argument("--vae-benchmark", type=Path, required=True)
    parser.add_argument("--diffusion-benchmark", type=Path, required=True)
    parser.add_argument(
        "--measurement-sequence-note",
        default=None,
        help="Factual note describing the order or reuse of the source measurements.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/generator_speed_comparison_2026-08-12",
    )
    args = parser.parse_args()
    summary = package_speed_report(
        args.vae_benchmark,
        args.diffusion_benchmark,
        args.report_dir,
        measurement_sequence_note=args.measurement_sequence_note,
    )
    print(f"report={args.report_dir.resolve()}")
    print(f"diffusion_to_vae_time_ratio={summary['comparison']['diffusion_to_vae_time_ratio']:.6f}")


if __name__ == "__main__":
    main()
