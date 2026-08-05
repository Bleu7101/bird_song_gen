from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bird_song.data import resolve_dataset_root
from bird_song.runtime import choose_device, save_json
from bird_song.vocoder import VocoderMelScaler, VocoderSpectrogramConfig, load_vocoder_waveform, waveform_to_vocoder_mel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the exact BigVGAN 80x256 raw log-mel cache.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--vocoder-config", type=Path, default=PROJECT_ROOT / "configs/bigvgan_spectrogram.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/bigvgan_mels")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    rows = pd.read_csv(args.manifest)
    required = {"split", "name", "filename", "relative_wav_path"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    if rows.empty or "train" not in set(rows["split"]):
        raise ValueError("the selected manifest rows must include training examples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = args.output_dir / "mel_manifest.csv"
    scaler_path = args.output_dir / "scaler.json"
    if (cache_manifest.exists() or scaler_path.exists()) and not args.overwrite:
        raise FileExistsError("cache metadata already exists; pass --overwrite")

    config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    device = choose_device(args.device)
    output_rows: list[dict[str, object]] = []
    minimum = float("inf")
    maximum = float("-inf")
    count = 0
    split_ranges: dict[str, dict[str, float | int]] = {}
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="BigVGAN log-mels"):
        relative_output = Path(str(row.split)) / slug(str(row.name)) / f"{Path(str(row.filename)).stem}.npy"
        destination = args.output_dir / relative_output
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"raw mel already exists: {destination}")
        waveform = load_vocoder_waveform(dataset_root / str(row.relative_wav_path), config).to(device)
        raw = waveform_to_vocoder_mel(waveform, config)[0].detach().cpu().numpy().astype(np.float32)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, raw, allow_pickle=False)
        if str(row.split) == "train":
            minimum = min(minimum, float(raw.min()))
            maximum = max(maximum, float(raw.max()))
            count += raw.size
        split = str(row.split)
        split_stats = split_ranges.setdefault(
            split,
            {"clips": 0, "values": 0, "minimum": float("inf"), "maximum": float("-inf")},
        )
        split_stats["clips"] = int(split_stats["clips"]) + 1
        split_stats["values"] = int(split_stats["values"]) + int(raw.size)
        split_stats["minimum"] = min(float(split_stats["minimum"]), float(raw.min()))
        split_stats["maximum"] = max(float(split_stats["maximum"]), float(raw.max()))
        record = row._asdict()
        record["relative_mel_path"] = relative_output.as_posix()
        output_rows.append(record)
    scaler = VocoderMelScaler(minimum=minimum, maximum=maximum, count=count)
    pd.DataFrame(output_rows).to_csv(cache_manifest, index=False)
    save_json(scaler_path, scaler.to_dict())
    save_json(args.output_dir / "vocoder_config.json", config.to_dict())
    for split_stats in split_ranges.values():
        split_stats["below_training_minimum"] = 0
        split_stats["above_training_maximum"] = 0
    for row in output_rows:
        raw = np.load(args.output_dir / str(row["relative_mel_path"]), allow_pickle=False)
        split_stats = split_ranges[str(row["split"])]
        split_stats["below_training_minimum"] = int(split_stats["below_training_minimum"]) + int((raw < scaler.minimum).sum())
        split_stats["above_training_maximum"] = int(split_stats["above_training_maximum"]) + int((raw > scaler.maximum).sum())
    save_json(
        args.output_dir / "cache_summary.json",
        {
            "rows": len(output_rows),
            "shape": [config.n_mels, config.expected_frames],
            "scaler": scaler.to_dict(),
            "split_ranges_and_clipping": split_ranges,
        },
    )
    print(f"Saved {len(output_rows)} raw mels with shape ({config.n_mels}, {config.expected_frames})")
    print(f"Training scaler: minimum={scaler.minimum:.6f}, maximum={scaler.maximum:.6f}, count={scaler.count}")
    for split, values in split_ranges.items():
        print(
            f"{split}: range=[{float(values['minimum']):.6f}, {float(values['maximum']):.6f}], "
            f"below_train_min={int(values['below_training_minimum'])}, "
            f"above_train_max={int(values['above_training_maximum'])}"
        )
    print(f"Cache manifest: {cache_manifest}")


if __name__ == "__main__":
    main()
