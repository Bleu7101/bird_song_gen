from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from bird_song.audio_evaluation import (
    embedding_frechet_distance,
    evaluate_audio_classifier,
    listening_median,
    multi_resolution_stft_error,
    select_balanced_recordings,
    waveform_diagnostics,
)
from bird_song.data import resolve_dataset_root
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-audio vocoder reconstruction gate.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_test.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--vocoder-config",
        type=Path,
        default=PROJECT_ROOT / "configs/vocoder_spectrogram.json",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "classifier_artifacts/Harvey_classifier/best.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/vocoder_gate")
    parser.add_argument("--per-species", type=int, default=30)
    parser.add_argument("--griffin-lim-iterations", type=int, default=32)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bigvgan-source", type=Path, default=None)
    parser.add_argument("--bigvgan-model", default=None)
    parser.add_argument("--listening-ratings", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.detach().cpu().numpy(), sample_rate, subtype="PCM_16")


def main() -> None:
    args = parse_args()
    if args.per_species < 1 or args.griffin_lim_iterations < 1:
        raise ValueError("per-species and Griffin-Lim iterations must be positive")
    summary_path = args.output_dir / "gate_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {summary_path}; pass --overwrite")
    seed_everything(args.seed)
    device = choose_device(args.device)
    config = VocoderSpectrogramConfig.from_json(args.vocoder_config)
    source = args.bigvgan_source or PROJECT_ROOT / "external" / config.model_id.rsplit("/", 1)[-1]
    model_reference: str | Path = args.bigvgan_model or (
        source if (source / "bigvgan_generator.pt").is_file() else config.model_id
    )
    vocoder = load_bigvgan(model_reference, device, config, source_dir=source)
    dataset_root = resolve_dataset_root(PROJECT_ROOT, args.dataset_root)
    manifest_rows = pd.read_csv(args.manifest)
    selected = select_balanced_recordings(manifest_rows, args.per_species, seed=args.seed)
    selected_paths = set(selected["relative_wav_path"])
    calibration_candidates = manifest_rows.loc[
        ~manifest_rows["relative_wav_path"].isin(selected_paths)
    ]
    real_calibration = select_balanced_recordings(
        calibration_candidates,
        args.per_species,
        seed=args.seed + 10_000,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict] = []
    paths_by_condition = {
        condition: []
        for condition in ("original", "real_calibration", "griffin_lim", "bigvgan")
    }
    targets: list[str] = []
    calibration_targets: list[str] = []
    diagnostics: list[dict] = []
    metric_rows: list[dict] = []
    for clip_index, row in enumerate(tqdm(selected.itertuples(index=False), total=len(selected), desc="Vocoder gate")):
        class_slug = slug(row.name)
        stem = f"clip_{clip_index:03d}_{Path(row.relative_wav_path).stem}"
        original = load_vocoder_waveform(dataset_root / row.relative_wav_path, config)
        raw_logmel = waveform_to_vocoder_mel(original.to(device, non_blocking=True), config)[0]
        griffin_lim = griffin_lim_from_vocoder_mel(
            raw_logmel,
            config,
            iterations=args.griffin_lim_iterations,
            seed=args.seed + clip_index,
        )
        bigvgan = vocoder_mel_to_waveform(raw_logmel, vocoder, config)
        condition_waveforms = {
            "original": original,
            "griffin_lim": griffin_lim,
            "bigvgan": bigvgan,
        }
        condition_paths = {}
        for condition, waveform in condition_waveforms.items():
            path = args.output_dir / "audio" / condition / class_slug / f"{stem}.wav"
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {path}")
            write_waveform(path, waveform, config.sample_rate)
            condition_paths[condition] = path
            paths_by_condition[condition].append(path)
            diagnostic = waveform_diagnostics(waveform, config.num_samples)
            diagnostics.append(
                {"clip_id": stem, "species": row.name, "condition": condition, **diagnostic}
            )
        for condition in ("griffin_lim", "bigvgan"):
            metric_rows.append(
                {
                    "clip_id": stem,
                    "species": row.name,
                    "condition": condition,
                    **multi_resolution_stft_error(original, condition_waveforms[condition]),
                }
            )
        output_rows.append(
            {
                "clip_id": stem,
                "species": row.name,
                "source_wav": str((dataset_root / row.relative_wav_path).resolve()),
                **{f"{condition}_wav": str(path.resolve()) for condition, path in condition_paths.items()},
            }
        )
        targets.append(row.name)

    calibration_rows = []
    for clip_index, row in enumerate(real_calibration.itertuples(index=False)):
        class_slug = slug(row.name)
        stem = f"real_calibration_{clip_index:03d}_{Path(row.relative_wav_path).stem}"
        waveform = load_vocoder_waveform(dataset_root / row.relative_wav_path, config)
        path = args.output_dir / "audio" / "real_calibration" / class_slug / f"{stem}.wav"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        write_waveform(path, waveform, config.sample_rate)
        paths_by_condition["real_calibration"].append(path)
        calibration_targets.append(row.name)
        diagnostics.append(
            {
                "clip_id": stem,
                "species": row.name,
                "condition": "real_calibration",
                **waveform_diagnostics(waveform, config.num_samples),
            }
        )
        calibration_rows.append(
            {
                "clip_id": stem,
                "species": row.name,
                "source_wav": str((dataset_root / row.relative_wav_path).resolve()),
                "real_calibration_wav": str(path.resolve()),
            }
        )

    condition_results = {}
    embeddings = {}
    for condition, paths in paths_by_condition.items():
        condition_targets = calibration_targets if condition == "real_calibration" else targets
        predictions, condition_embeddings, accuracy = evaluate_audio_classifier(
            args.classifier_checkpoint,
            paths,
            condition_targets,
            device,
            batch_size=args.classifier_batch_size,
            workers=args.workers,
        )
        predictions.insert(0, "condition", condition)
        predictions.to_csv(args.output_dir / f"classifier_{condition}.csv", index=False)
        embeddings[condition] = condition_embeddings
        condition_results[condition] = {"accuracy": accuracy, "samples": len(paths)}

    diagnostics_frame = pd.DataFrame(diagnostics)
    metrics_frame = pd.DataFrame(metric_rows)
    diagnostics_frame.to_csv(args.output_dir / "waveform_diagnostics.csv", index=False)
    metrics_frame.to_csv(args.output_dir / "paired_audio_metrics.csv", index=False)
    pd.DataFrame(output_rows).to_csv(args.output_dir / "evaluation_manifest.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        args.output_dir / "real_calibration_manifest.csv", index=False
    )
    all_valid = bool(diagnostics_frame["valid"].all())
    max_clipped_fraction = float(diagnostics_frame["clipped_fraction"].max())
    accuracy_drop = condition_results["original"]["accuracy"] - condition_results["bigvgan"]["accuracy"]
    automatic_pass = all_valid and max_clipped_fraction <= 0.001 and accuracy_drop <= 0.05
    bird_likeness = None
    if args.listening_ratings is not None:
        bird_likeness = listening_median(args.listening_ratings, "bigvgan")
    if not automatic_pass:
        gate_status = "failed_automatic"
    elif bird_likeness is None:
        gate_status = "awaiting_listening"
    elif bird_likeness >= 3.0:
        gate_status = "passed"
    else:
        gate_status = "failed_listening"

    response_template = pd.DataFrame(
        {
            "rater_id": "",
            "clip_id": [row["clip_id"] for row in output_rows],
            "condition": "bigvgan",
            "bird_likeness": "",
            "notes": "",
        }
    )
    response_template.to_csv(args.output_dir / "pilot_listening_response_template.csv", index=False)
    summary = {
        "gate_status": gate_status,
        "automatic_pass": automatic_pass,
        "criteria": {
            "maximum_classifier_accuracy_drop": 0.05,
            "minimum_listening_median": 3.0,
            "maximum_clipped_fraction": 0.001,
        },
        "classifier": condition_results,
        "bigvgan_accuracy_drop": accuracy_drop,
        "listening_bird_likeness_median": bird_likeness,
        "all_waveforms_valid": all_valid,
        "maximum_clipped_fraction": max_clipped_fraction,
        "paired_metrics_mean": metrics_frame.groupby("condition")[
            ["mrstft_spectral_convergence", "mrstft_log_magnitude_l1"]
        ].mean().to_dict(orient="index"),
        "classifier_embedding_frechet": {
            "real_vs_real": embedding_frechet_distance(
                embeddings["original"], embeddings["real_calibration"]
            ),
            **{
                condition: embedding_frechet_distance(
                    embeddings["original"], embeddings[condition]
                )
                for condition in ("griffin_lim", "bigvgan")
            },
        },
        "embedding_metric_note": (
            "FAD-style Gaussian Frechet distance in the published residual classifier embedding space; "
            "this is not the standard VGGish FAD and is reported only as a domain-specific diagnostic."
        ),
        "vocoder_config": config.to_dict(),
        "selected_samples": len(selected),
        "real_calibration_samples": len(real_calibration),
    }
    save_json(summary_path, summary)
    print(f"Gate status: {gate_status}")
    print(f"Original classifier accuracy: {condition_results['original']['accuracy']:.2%}")
    print(f"BigVGAN classifier accuracy: {condition_results['bigvgan']['accuracy']:.2%}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
