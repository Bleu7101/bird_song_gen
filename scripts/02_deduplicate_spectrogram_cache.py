from __future__ import annotations

import argparse
import json
from pathlib import Path

from bird_song.config import SpectrogramConfig
from bird_song.spectrogram_cache import canonicalize_spectrogram_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit or canonicalize the real-spectrogram cache by array content.")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "artifacts/spectrograms")
    parser.add_argument("--spectrogram-config", type=Path, default=PROJECT_ROOT / "configs/spectrogram.json")
    parser.add_argument("--apply", action="store_true", help="Create content-addressed objects and remove redundant files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.spectrogram_config.resolve()
    result = canonicalize_spectrogram_cache(
        args.cache_root,
        SpectrogramConfig.from_json(config_path),
        config_path,
        apply=args.apply,
    )
    print(json.dumps(result, indent=2))
    if not args.apply:
        print("Dry run only; pass --apply to write the canonical cache layout.")


if __name__ == "__main__":
    main()
