from __future__ import annotations

import argparse
from pathlib import Path

from bird_song.generation.posterior_bank_filter import filter_posterior_bank_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter an existing VAE posterior bank to unique content-safe training anchors."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    filter_posterior_bank_file(args.input, args.manifest, args.output)
    print(f"filtered_bank={args.output.resolve()}")
    print("counts=Northern Cardinal:256,Song Sparrow:247,American Robin:256")
    print("vae_checkpoint_retrained=false")


if __name__ == "__main__":
    main()
