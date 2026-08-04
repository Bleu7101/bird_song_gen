from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only the frozen BigVGAN generator and source files required for inference."
    )
    parser.add_argument(
        "--model-id",
        default="nvidia/bigvgan_v2_22khz_80band_256x",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required; install requirements.txt in the project environment"
        ) from error

    destination = args.output_dir or PROJECT_ROOT / "external" / args.model_id.rsplit("/", 1)[-1]
    required = [destination / "config.json", destination / "bigvgan_generator.pt", destination / "bigvgan.py"]
    if all(path.is_file() for path in required) and not args.force:
        print(f"BigVGAN files already exist in {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=args.model_id,
        local_dir=destination,
        allow_patterns=[
            "config.json",
            "bigvgan_generator.pt",
            "*.py",
            "alias_free_activation/**",
            "LICENSE",
            "incl_licenses/**",
        ],
        force_download=args.force,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Download completed but required files are missing: {missing}")
    print(f"Frozen BigVGAN generator: {downloaded}")
    print("Use this directory for both --bigvgan-source and --bigvgan-model.")


if __name__ == "__main__":
    main()
