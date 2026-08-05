from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from bird_song.data import resolve_dataset_root
from bird_song.metrics import multi_resolution_stft_error, waveform_diagnostics
from bird_song.runtime import choose_device, save_json, seed_everything
from bird_song.vocoder import (
    VocoderSpectrogramConfig,
    griffin_lim_from_vocoder_mel,
    load_bigvgan,
    load_vocoder_waveform,
    vocoder_mel_to_waveform,
    waveform_to_vocoder_mel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.detach().cpu().numpy(), sample_rate, subtype="PCM_16")


def balanced_rows(rows: pd.DataFrame, per_species: int, seed: int) -> pd.DataFrame:
    if per_species < 1:
        raise ValueError("per-species must be positive")
    rng = torch.Generator().manual_seed(seed)
    selected = []
    for name in ("American Robin", "Northern Cardinal", "Song Sparrow"):
        group = rows.loc[rows["name"].astype(str).eq(name)].reset_index(drop=True)
        if len(group) < per_species:
            raise ValueError(f"only {len(group)} rows available for {name}")
        indices = torch.randperm(len(group), generator=rng)[:per_species].tolist()
        selected.append(group.iloc[indices])
    return pd.concat(selected, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BigVGAN and exact-mel Griffin-Lim on held-out real clips.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--vocoder-config", type=Path, default=PROJECT_ROOT / "configs/bigvgan_spectrogram.json")
    parser.add_argument("--bigvgan-source", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/bigvgan_real_test")
    parser.add_argument("--per-species", type=int, default=30)
    parser.add_argument("--griffin-lim-iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    device = choose_device(args.device)
    source = args.bigvgan_source or PROJECT_ROOT / "external" / config.model_id.rsplit("/", 1)[-1]
    model = load_bigvgan(source, device, config)
    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    manifest = pd.read_csv(args.manifest)
    selected = balanced_rows(manifest.loc[manifest["split"].astype(str).eq("test")], args.per_species, args.seed)
    summary_path = args.output_dir / "decoder_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {summary_path}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, row in enumerate(tqdm(selected.itertuples(index=False), total=len(selected), desc="BigVGAN decoder test")):
        species = str(row.name)
        stem = f"clip_{index:03d}_{Path(str(row.relative_wav_path)).stem}"
        original = load_vocoder_waveform(dataset_root / str(row.relative_wav_path), config)
        raw_mel = waveform_to_vocoder_mel(original.to(device), config)[0]
        griffin = griffin_lim_from_vocoder_mel(raw_mel, config, args.griffin_lim_iterations)
        bigvgan = vocoder_mel_to_waveform(raw_mel, model, config)
        paths = {}
        for condition, waveform in (("original", original), ("griffin_lim", griffin), ("bigvgan", bigvgan)):
            path = args.output_dir / "audio" / condition / slug(species) / f"{stem}.wav"
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {path}")
            write_wav(path, waveform, config.sample_rate)
            paths[condition] = str(path.relative_to(args.output_dir).as_posix())
        bigvgan_metrics = multi_resolution_stft_error(original, bigvgan)
        griffin_metrics = multi_resolution_stft_error(original, griffin)
        records.append(
            {
                "clip_id": stem,
                "species": species,
                "source_wav": str(row.relative_wav_path),
                "original_wav": paths["original"],
                "griffin_lim_wav": paths["griffin_lim"],
                "bigvgan_wav": paths["bigvgan"],
                **{f"griffin_lim_{key}": value for key, value in griffin_metrics.items()},
                **{f"bigvgan_{key}": value for key, value in bigvgan_metrics.items()},
                **{f"bigvgan_{key}": value for key, value in waveform_diagnostics(bigvgan, config.num_samples).items()},
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "decoder_metrics.csv", index=False)
    frame[["clip_id", "species", "original_wav", "griffin_lim_wav", "bigvgan_wav"]].to_csv(
        args.output_dir / "listening_manifest.csv", index=False
    )
    valid = bool((frame["bigvgan_valid"] == 1).all())
    max_clipped = float(frame["bigvgan_clipped_fraction"].max())
    griffin_sc = float(frame["griffin_lim_mrstft_spectral_convergence"].mean())
    bigvgan_sc = float(frame["bigvgan_mrstft_spectral_convergence"].mean())
    griffin_l1 = float(frame["griffin_lim_mrstft_log_magnitude_l1"].mean())
    bigvgan_l1 = float(frame["bigvgan_mrstft_log_magnitude_l1"].mean())
    automatic_pass = valid and max_clipped <= 0.001 and bigvgan_sc < griffin_sc and bigvgan_l1 < griffin_l1
    summary = {
        "status": "automatic_pass" if automatic_pass else "automatic_fail",
        "automatic_pass": automatic_pass,
        "selected_samples": len(frame),
        "per_species": args.per_species,
        "config": config.to_dict(),
        "device": str(device),
        "all_bigvgan_waveforms_valid": valid,
        "maximum_bigvgan_clipped_fraction": max_clipped,
        "mean_metrics": {
            "griffin_lim_mrstft_spectral_convergence": griffin_sc,
            "bigvgan_mrstft_spectral_convergence": bigvgan_sc,
            "griffin_lim_mrstft_log_magnitude_l1": griffin_l1,
            "bigvgan_mrstft_log_magnitude_l1": bigvgan_l1,
        },
        "note": "Automatic metrics do not replace listening; generated-sample realism is evaluated separately.",
    }
    save_json(summary_path, summary)
    print(f"Decoder status: {summary['status']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
