from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
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
    VocoderMelNormalizer,
    VocoderSpectrogramConfig,
    load_bigvgan,
    load_vocoder_waveform,
    vocoder_mel_to_waveform,
    waveform_to_vocoder_mel,
)
from bird_song.vocoder_vae.model import ConditionalVocoderVAE, VocoderVAEConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate vocoder ceiling, deterministic VAE reconstructions, and VAE audio samples."
    )
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "runs/vocoder_vae/best.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_test.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--generated-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/vocoder_vae/generated_manifest.csv",
    )
    parser.add_argument("--generated-root", type=Path, default=PROJECT_ROOT / "outputs/vocoder_vae")
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "classifier_artifacts/Harvey_classifier/best.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/vocoder_vae/evaluation")
    parser.add_argument("--per-species", type=int, default=30)
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


def load_generated_rows(
    manifest_path: Path,
    generated_root: Path,
    classes: tuple[str, ...],
    per_species: int,
    seed: int,
) -> tuple[list[Path], list[str]]:
    rows = pd.read_csv(manifest_path)
    required = {"species", "audio_path"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Generated manifest is missing columns: {sorted(missing)}")
    rows = rows.loc[rows["audio_path"].fillna("").astype(str).str.len() > 0].copy()
    unknown = sorted(set(rows["species"]) - set(classes))
    if unknown:
        raise ValueError(f"Generated manifest contains unknown classes: {unknown}")
    selected = []
    for class_index, class_name in enumerate(classes):
        candidates = rows.loc[rows["species"] == class_name]
        if len(candidates) < per_species:
            raise ValueError(
                f"Generated manifest has {len(candidates)} audio samples for {class_name!r}; "
                f"need {per_species}"
            )
        selected.append(candidates.sample(per_species, random_state=seed + class_index))
    frame = pd.concat(selected).reset_index(drop=True)
    paths = [(generated_root / Path(value)).resolve() for value in frame["audio_path"]]
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Generated audio files are missing, first example: {missing_paths[0]}")
    return paths, frame["species"].tolist()


def validate_saved_waveforms(
    paths: list[Path],
    targets: list[str],
    condition: str,
    config: VocoderSpectrogramConfig,
) -> list[dict]:
    rows = []
    for path, target in zip(paths, targets):
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy()).mean(0)
        diagnostic = waveform_diagnostics(waveform, config.num_samples)
        rows.append(
            {
                "path": str(path),
                "species": target,
                "condition": condition,
                "sample_rate": sample_rate,
                **diagnostic,
                "valid": bool(diagnostic["valid"] and sample_rate == config.sample_rate),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.per_species < 1 or args.classifier_batch_size < 1 or args.workers < 0:
        raise ValueError("per-species and classifier batch size must be positive; workers cannot be negative")
    summary_path = args.output_dir / "audio_gate_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {summary_path}; pass --overwrite")

    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = VocoderVAEConfig.from_dict(checkpoint["model_config"])
    vocoder_config = VocoderSpectrogramConfig.from_dict(checkpoint["vocoder_config"])
    normalizer = VocoderMelNormalizer.from_dict(checkpoint["normalizer"])
    classes = tuple(checkpoint["classes"])
    model = ConditionalVocoderVAE(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    default_source = PROJECT_ROOT / "external" / vocoder_config.model_id.rsplit("/", 1)[-1]
    source = args.bigvgan_source or default_source
    model_reference: str | Path = args.bigvgan_model or (
        source if (source / "bigvgan_generator.pt").is_file() else vocoder_config.model_id
    )
    vocoder = load_bigvgan(model_reference, device, vocoder_config, source_dir=source)
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

    paths_by_condition = {
        condition: []
        for condition in ("original", "real_calibration", "bigvgan", "vae_reconstruction")
    }
    reconstruction_targets: list[str] = []
    calibration_targets: list[str] = []
    evaluation_rows: list[dict] = []
    diagnostics: list[dict] = []
    metric_rows: list[dict] = []
    class_to_index = {name: index for index, name in enumerate(classes)}

    with torch.inference_mode():
        for clip_index, row in enumerate(
            tqdm(selected.itertuples(index=False), total=len(selected), desc="VAE audio evaluation")
        ):
            if row.name not in class_to_index:
                raise ValueError(f"VAE checkpoint does not contain class {row.name!r}")
            class_slug = slug(row.name)
            stem = f"clip_{clip_index:03d}_{Path(row.relative_wav_path).stem}"
            original = load_vocoder_waveform(dataset_root / row.relative_wav_path, vocoder_config)
            raw_logmel = waveform_to_vocoder_mel(
                original.to(device, non_blocking=True), vocoder_config
            )[0]
            normalized = normalizer.normalize(raw_logmel)[None, None].to(device)
            labels = torch.tensor([class_to_index[row.name]], device=device)
            normalized_reconstruction, _, _ = model(normalized, labels, sample_latent=False)
            raw_reconstruction = normalizer.denormalize(normalized_reconstruction)[0, 0]
            ceiling = vocoder_mel_to_waveform(raw_logmel, vocoder, vocoder_config)
            vae_reconstruction = vocoder_mel_to_waveform(raw_reconstruction, vocoder, vocoder_config)
            waveforms = {
                "original": original,
                "bigvgan": ceiling,
                "vae_reconstruction": vae_reconstruction,
            }
            condition_paths = {}
            for condition, waveform in waveforms.items():
                path = args.output_dir / "audio" / condition / class_slug / f"{stem}.wav"
                if path.exists() and not args.overwrite:
                    raise FileExistsError(f"Refusing to overwrite {path}")
                write_waveform(path, waveform, vocoder_config.sample_rate)
                paths_by_condition[condition].append(path)
                condition_paths[condition] = path
                diagnostics.append(
                    {
                        "clip_id": stem,
                        "species": row.name,
                        "condition": condition,
                        **waveform_diagnostics(waveform, vocoder_config.num_samples),
                    }
                )
            for condition in ("bigvgan", "vae_reconstruction"):
                metric_rows.append(
                    {
                        "clip_id": stem,
                        "species": row.name,
                        "condition": condition,
                        **multi_resolution_stft_error(original, waveforms[condition]),
                    }
                )
            evaluation_rows.append(
                {
                    "clip_id": stem,
                    "species": row.name,
                    "source_wav": str((dataset_root / row.relative_wav_path).resolve()),
                    **{
                        f"{condition}_wav": str(path.resolve())
                        for condition, path in condition_paths.items()
                    },
                }
            )
            reconstruction_targets.append(row.name)

    calibration_rows = []
    for clip_index, row in enumerate(real_calibration.itertuples(index=False)):
        class_slug = slug(row.name)
        stem = f"real_calibration_{clip_index:03d}_{Path(row.relative_wav_path).stem}"
        waveform = load_vocoder_waveform(dataset_root / row.relative_wav_path, vocoder_config)
        path = args.output_dir / "audio" / "real_calibration" / class_slug / f"{stem}.wav"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        write_waveform(path, waveform, vocoder_config.sample_rate)
        paths_by_condition["real_calibration"].append(path)
        calibration_targets.append(row.name)
        diagnostics.append(
            {
                "clip_id": stem,
                "species": row.name,
                "condition": "real_calibration",
                **waveform_diagnostics(waveform, vocoder_config.num_samples),
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

    generated_paths, generated_targets = load_generated_rows(
        args.generated_manifest,
        args.generated_root,
        classes,
        args.per_species,
        args.seed,
    )
    paths_by_condition["vae_generated"] = generated_paths
    diagnostics.extend(
        validate_saved_waveforms(
            generated_paths,
            generated_targets,
            "vae_generated",
            vocoder_config,
        )
    )

    classifier_results = {}
    embeddings = {}
    for condition, paths in paths_by_condition.items():
        if condition == "vae_generated":
            targets = generated_targets
        elif condition == "real_calibration":
            targets = calibration_targets
        else:
            targets = reconstruction_targets
        predictions, condition_embeddings, accuracy = evaluate_audio_classifier(
            args.classifier_checkpoint,
            paths,
            targets,
            device,
            batch_size=args.classifier_batch_size,
            workers=args.workers,
        )
        predictions.insert(0, "condition", condition)
        predictions.to_csv(args.output_dir / f"classifier_{condition}.csv", index=False)
        embeddings[condition] = condition_embeddings
        classifier_results[condition] = {"accuracy": accuracy, "samples": len(paths)}

    diagnostics_frame = pd.DataFrame(diagnostics)
    metrics_frame = pd.DataFrame(metric_rows)
    diagnostics_frame.to_csv(args.output_dir / "waveform_diagnostics.csv", index=False)
    metrics_frame.to_csv(args.output_dir / "paired_audio_metrics.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(args.output_dir / "reconstruction_manifest.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        args.output_dir / "real_calibration_manifest.csv", index=False
    )

    all_valid = bool(diagnostics_frame["valid"].all())
    maximum_clipped_fraction = float(diagnostics_frame["clipped_fraction"].max())
    reconstruction_drop = (
        classifier_results["bigvgan"]["accuracy"]
        - classifier_results["vae_reconstruction"]["accuracy"]
    )
    generated_accuracy = classifier_results["vae_generated"]["accuracy"]
    automatic_pass = (
        all_valid
        and maximum_clipped_fraction <= 0.001
        and reconstruction_drop <= 0.10
        and generated_accuracy > 0.50
    )
    bird_likeness = None
    if args.listening_ratings is not None:
        bird_likeness = listening_median(args.listening_ratings, "vae_generated")
    if not automatic_pass:
        gate_status = "failed_automatic"
    elif bird_likeness is None:
        gate_status = "awaiting_listening"
    elif bird_likeness >= 3.0:
        gate_status = "audio_ready"
    else:
        gate_status = "failed_listening"

    summary = {
        "gate_status": gate_status,
        "automatic_pass": automatic_pass,
        "criteria": {
            "maximum_reconstruction_classifier_drop": 0.10,
            "minimum_generated_target_accuracy_exclusive": 0.50,
            "minimum_generated_bird_likeness_median": 3.0,
            "maximum_clipped_fraction": 0.001,
        },
        "classifier": classifier_results,
        "vae_reconstruction_accuracy_drop_from_vocoder_ceiling": reconstruction_drop,
        "generated_bird_likeness_median": bird_likeness,
        "all_waveforms_valid": all_valid,
        "maximum_clipped_fraction": maximum_clipped_fraction,
        "paired_metrics_mean": metrics_frame.groupby("condition")[[
            "mrstft_spectral_convergence",
            "mrstft_log_magnitude_l1",
        ]].mean().to_dict(orient="index"),
        "classifier_embedding_frechet": {
            "real_vs_real": embedding_frechet_distance(
                embeddings["original"], embeddings["real_calibration"]
            ),
            **{
                condition: embedding_frechet_distance(
                    embeddings["original"], embeddings[condition]
                )
                for condition in ("bigvgan", "vae_reconstruction", "vae_generated")
            },
        },
        "embedding_metric_note": (
            "FAD-style Gaussian Frechet distance in the published residual classifier embedding space; "
            "this is not standard VGGish FAD and is never used alone."
        ),
        "vocoder_config": vocoder_config.to_dict(),
        "checkpoint_epoch": checkpoint["epoch"],
        "selected_samples_per_condition": args.per_species * len(classes),
        "real_calibration_samples": len(real_calibration),
    }
    save_json(summary_path, summary)
    print(f"VAE audio gate status: {gate_status}")
    print(f"Vocoder ceiling classifier accuracy: {classifier_results['bigvgan']['accuracy']:.2%}")
    print(f"VAE reconstruction classifier accuracy: {classifier_results['vae_reconstruction']['accuracy']:.2%}")
    print(f"VAE generated classifier accuracy: {generated_accuracy:.2%}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
