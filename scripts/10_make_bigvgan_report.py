from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["American Robin", "Northern Cardinal", "Song Sparrow"]
CONTRACT = {
    "sample_rate": 22050,
    "num_samples": 65536,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 80,
    "expected_frames": 256,
    "model_id": "nvidia/bigvgan_v2_22khz_80band_256x",
    "representation": "raw natural-log mel -> global train-only min-max scaled to [-1, 1]",
    "classes": CLASSES,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def history_best(path: Path, metric: str, lower_is_better: bool = True) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best = min(rows, key=lambda row: float(row[metric])) if lower_is_better else max(rows, key=lambda row: float(row[metric]))
    return {"epoch": int(best["epoch"]), metric: float(best[metric]), "epochs_recorded": len(rows)}


def generation_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    diagnostics = summary.get("mel_diagnostics", {})
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "label": path.parent.relative_to(PROJECT_ROOT).as_posix(),
        "status": summary.get("status"),
        "generated_samples": summary.get("generated_samples"),
        "samples_per_species": summary.get("samples_per_species"),
        "temperature": summary.get("temperature", ""),
        "seed": summary.get("seed"),
        "maximum_clipped_fraction": summary.get("maximum_clipped_fraction"),
        "silent_samples": summary.get("silent_samples"),
        "mean_scaled_saturation_fraction": diagnostics.get("mean_scaled_saturation_fraction"),
        "mean_pairwise_scaled_l2_distance": diagnostics.get("mean_pairwise_scaled_l2_distance"),
    }


def classifier_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    result: dict[str, Any] = {"path": path.relative_to(PROJECT_ROOT).as_posix(), "classifiers": {}}
    for name, values in summary.get("classifiers", {}).items():
        result["classifiers"][name] = {
            "num_samples": values.get("num_samples"),
            "mean_confidence": values.get("mean_confidence"),
            "target_label_accuracy": values.get("target_label_accuracy"),
            "per_target_accuracy": values.get("per_target_accuracy"),
        }
    return result


def make_markdown(report: dict[str, Any]) -> str:
    selection = report["selection"]
    lines = [
        "# BigVGAN generator retraining evidence",
        "",
        "This report was generated from the recorded `decoder_test` worktree. Checkpoints are selected from validation evidence; decoded audio and frozen-classifier results are separate diagnostics.",
        "",
        "## Contract",
        "",
        f"- BigVGAN: `{CONTRACT['model_id']}`; {CONTRACT['sample_rate']} Hz; {CONTRACT['num_samples']} samples; raw log-mels `{CONTRACT['n_mels']}×{CONTRACT['expected_frames']}`.",
        f"- Scaler: `{report['scaler']}`.",
        f"- Classes: {', '.join(CLASSES)}.",
        "",
        "## Validation-only matrix",
        "",
        "| Run | Seed | Best epoch | Validation metric |",
        "|---|---:|---:|---:|",
    ]
    for row in report["matrix"]:
        metric = row["best_validation_nll"] if "best_validation_nll" in row else row["best_selection_error"]
        lines.append(f"| `{row['run']}` | {row['seed']} | {row['best_epoch']} | {metric:.6f} |" )
    lines.extend(
        [
            "",
            f"WGAN selection: `{selection['wgan']['config']}` (mean error {selection['wgan']['mean']:.6f}, median {selection['wgan']['median']:.6f}); selected seed {selection['wgan']['seed']}.",
            f"Transformer selection: `{selection['transformer']['patches']}` (mean validation NLL {selection['transformer']['mean']:.6f}, median {selection['transformer']['median']:.6f}); selected seed {selection['transformer']['seed']}.",
            "",
            "## Output gates",
            "",
            "All recorded balanced sets below have finite 80×256 mels, valid 65,536-sample waveforms, zero silent samples, and clipped fraction below 0.001. The temperature sweep is reported for listening choice; it is not a validation-loss selection criterion.",
            "",
            "| Set | Status | Samples | Temperature | Max clipping | Saturation | Diversity |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["generation"]:
        lines.append(
            f"| `{row['label']}` | {row['status']} | {row['generated_samples']} | {row.get('temperature', '')} | {row['maximum_clipped_fraction']:.6f} | {row['mean_scaled_saturation_fraction']:.6f} | {row['mean_pairwise_scaled_l2_distance']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Conditioning diagnostics",
            "",
            "The frozen legacy residual CNN and CRNN were run on the generated BigVGAN WAVs only. Their agreement is a conditioning diagnostic, not a perceptual realism score and is not mixed into validation selection.",
            "",
            "| Diagnostic set | Residual CNN accuracy | CRNN accuracy |",
            "|---|---:|---:|",
        ]
    )
    for row in report["classifier_diagnostics"]:
        values = row["classifiers"]
        lines.append(
            f"| `{Path(row['path']).parent.as_posix()}` | {values.get('residual_cnn', {}).get('target_label_accuracy', float('nan')):.3f} | {values.get('crnn', {}).get('target_label_accuracy', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Selected artifacts",
            "",
            "The publication bundle contains only the selected generator checkpoints, configs, hashes, model cards, and a small balanced WAV set. Downloaded BigVGAN weights, resumable checkpoints, caches, bulk arrays, and bulk audio remain excluded.",
            "",
        ]
    )
    for item in report["artifacts"]:
        lines.append(f"- `{item['path']}` — SHA-256 `{item['sha256']}` ({item['bytes']} bytes).")
    lines.extend(["", "## Caveat", "", "Automatic gates establish a compatible and finite decoder handoff. They do not establish perceptual realism; listen to the curated WAVs and treat the classifier outputs as diagnostics only.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact, source-backed BigVGAN matrix evidence.")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports/bigvgan_matrix_report.json")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "reports/bigvgan_matrix_report.md")
    args = parser.parse_args()

    matrix: list[dict[str, Any]] = []
    for run_dir in sorted((PROJECT_ROOT / "runs/experiments").iterdir()):
        if not run_dir.is_dir() or "seed" not in run_dir.name:
            continue
        seed = int(run_dir.name.rsplit("seed", 1)[1])
        best_path = run_dir / ("best_generator.pt" if run_dir.name.startswith("wgan_") else "best.pt")
        values = checkpoint(best_path)
        row: dict[str, Any] = {"run": run_dir.name, "seed": seed, "best_epoch": int(values["epoch"]), "checkpoint": best_path.relative_to(PROJECT_ROOT).as_posix()}
        if run_dir.name.startswith("wgan_"):
            row["configuration"] = "stability" if "stability" in run_dir.name else "current"
            row["best_selection_error"] = float(values["metrics"]["selection_error"])
        else:
            row["patches"] = values["model_config"]["patch_height"]
            row["best_validation_nll"] = float(values["metrics"]["validation_nll"])
        matrix.append(row)

    wgan = [row for row in matrix if "best_selection_error" in row]
    transformer = [row for row in matrix if "best_validation_nll" in row]
    wgan_groups: dict[str, list[float]] = {name: [row["best_selection_error"] for row in wgan if row["configuration"] == name] for name in ("current", "stability")}
    transformer_groups: dict[int, list[float]] = {patches: [row["best_validation_nll"] for row in transformer if row["patches"] == patches] for patches in (16, 8)}
    winning_wgan = min(wgan_groups, key=lambda name: statistics.mean(wgan_groups[name]))
    winning_transformer = min(transformer_groups, key=lambda patches: statistics.mean(transformer_groups[patches]))
    selected_wgan = min((row for row in wgan if row["configuration"] == winning_wgan), key=lambda row: row["best_selection_error"])
    selected_transformer = min((row for row in transformer if row["patches"] == winning_transformer), key=lambda row: row["best_validation_nll"])

    scaler = load_json(PROJECT_ROOT / "artifacts/bigvgan_mels/scaler.json")
    generation = [generation_summary(path) for path in sorted((PROJECT_ROOT / "runs/final_evaluations").rglob("generation_summary.json"))]
    diagnostics = [classifier_summary(path) for path in sorted((PROJECT_ROOT / "reports/legacy_classifier_diagnostics").glob("*/summary.json"))]
    artifacts = []
    for path in sorted((PROJECT_ROOT / "reports/canonical_models").glob("*")):
        if path.is_file() and path.suffix in {".pt", ".wav"}:
            artifacts.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})

    report = {
        "generated_from": "decoder_test",
        "contract": CONTRACT,
        "scaler": scaler,
        "matrix": matrix,
        "selection": {
            "wgan": {"config": winning_wgan, "mean": statistics.mean(wgan_groups[winning_wgan]), "median": statistics.median(wgan_groups[winning_wgan]), "seed": selected_wgan["seed"], "run": selected_wgan["run"]},
            "transformer": {"patches": f"{winning_transformer}x16", "mean": statistics.mean(transformer_groups[winning_transformer]), "median": statistics.median(transformer_groups[winning_transformer]), "seed": selected_transformer["seed"], "run": selected_transformer["run"]},
        },
        "generation": generation,
        "classifier_diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(make_markdown(report), encoding="utf-8")
    print(f"Saved {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
