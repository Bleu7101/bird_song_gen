from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the frozen BigVGAN generator and inference source.")
    parser.add_argument("--model-id", default="nvidia/bigvgan_v2_22khz_80band_256x")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = args.output_dir or PROJECT_ROOT / "external" / args.model_id.rsplit("/", 1)[-1]
    required = [destination / "config.json", destination / "bigvgan.py", destination / "bigvgan_generator.pt"]
    if all(path.is_file() for path in required) and not args.force:
        print(f"BigVGAN already available at {destination}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before fetching BigVGAN") from error
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        local_dir=destination,
        allow_patterns=["config.json", "bigvgan_generator.pt", "*.py", "alias_free_activation/**", "LICENSE"],
        force_download=args.force,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"BigVGAN download is incomplete: {missing}")
    print(f"Fetched BigVGAN to {destination}")


if __name__ == "__main__":
    main()
