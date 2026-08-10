from __future__ import annotations

import argparse
from pathlib import Path

from bird_song.generation.checkpoint_pool import generate_pool
from bird_song.runtime import choose_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a resumable classifier-input pool from a frozen VAE-v3 or diffusion checkpoint."
    )
    parser.add_argument("--model", choices=("vae_v3", "diffusion"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples-per-species", type=int, default=200)
    parser.add_argument("--posterior-bank", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Pool directory (default: runs/generator_checkpoint_evaluation/pools/<model>/seed_<seed>).",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--expected-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    output = args.output or PROJECT_ROOT / "runs/generator_checkpoint_evaluation/pools" / args.model / f"seed_{args.seed}"
    manifest = generate_pool(
        model=args.model,
        checkpoint=args.checkpoint,
        seed=args.seed,
        samples_per_species=args.samples_per_species,
        output=output,
        device=device,
        posterior_bank=args.posterior_bank,
        chunk_size=args.chunk_size,
        expected_sha256=args.expected_sha256,
    )
    print(f"pool={manifest}")


if __name__ == "__main__":
    main()
