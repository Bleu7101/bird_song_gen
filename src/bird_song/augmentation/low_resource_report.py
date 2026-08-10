from __future__ import annotations

import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from bird_song.augmentation.low_resource import GENERATORS, _portable, pool_reference
from bird_song.runtime import save_json
from bird_song.spectrogram_cache import sha256_file


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty_worktree_at_packaging": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty_worktree_at_packaging": None}


def _write_report_charts(
    report_dir: Path,
    validation_selection: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    block_count: int,
) -> None:
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for generator, group in validation_selection.groupby("generator"):
        ordered = group.sort_values("ratio_per_species")
        axis.errorbar(
            ordered["ratio_per_species"],
            ordered["validation_macro_f1_mean"],
            yerr=ordered["validation_macro_f1_sample_sd"],
            marker="o",
            capsize=4,
            label={"vae_v3": "VAE-v3", "diffusion": "Diffusion"}.get(generator, generator),
        )
    lower = float(
        (validation_selection["validation_macro_f1_mean"] - validation_selection["validation_macro_f1_sample_sd"]).min()
    )
    upper = float(
        (validation_selection["validation_macro_f1_mean"] + validation_selection["validation_macro_f1_sample_sd"]).max()
    )
    axis.set(
        xlabel="Generated spectrograms per species",
        ylabel="Validation macro F1",
        ylim=(max(0.0, lower - 0.02), min(1.0, upper + 0.02)),
        title=f"Low-resource validation selection across {block_count} matched blocks",
    )
    axis.legend(title="Generator")
    fig.savefig(figure_dir / "validation_selection.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    generators = list(GENERATORS)
    positions = np.arange(len(generators))
    values = [
        paired.loc[paired["generator"] == generator, "delta_macro_f1"].to_numpy()
        for generator in generators
    ]
    axis.axhline(0, color="black", linewidth=1)
    axis.boxplot(values, positions=positions, widths=0.5, showmeans=True)
    for position, group_values in zip(positions, values, strict=True):
        axis.scatter(np.full(len(group_values), position), group_values, color="#3366aa", zorder=3)
    axis.set(
        xticks=positions,
        xticklabels=["VAE-v3", "Diffusion"],
        ylabel="Paired test macro-F1 delta vs real-only",
        title="Synthetic augmentation effect in each matched block",
    )
    fig.savefig(figure_dir / "paired_test_deltas.png", dpi=160)
    plt.close(fig)


def _write_checksums(report_dir: Path) -> None:
    lines = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{sha256_file(path)}  {path.relative_to(report_dir).as_posix()}")
    (report_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subset_overlap_table(membership: pd.DataFrame) -> pd.DataFrame:
    required = {"low_resource_subset_seed", "name", "id"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(f"Subset membership is missing overlap columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    seeds = sorted(int(value) for value in membership["low_resource_subset_seed"].unique())
    for species, species_rows in membership.groupby("name", sort=True):
        ids_by_seed = {
            int(seed): set(group["id"].astype(str))
            for seed, group in species_rows.groupby("low_resource_subset_seed")
        }
        for seed_a, seed_b in combinations(seeds, 2):
            shared = ids_by_seed[seed_a] & ids_by_seed[seed_b]
            union = ids_by_seed[seed_a] | ids_by_seed[seed_b]
            rows.append(
                {
                    "species": species,
                    "subset_seed_a": seed_a,
                    "subset_seed_b": seed_b,
                    "shared_recording_ids": len(shared),
                    "union_recording_ids": len(union),
                    "jaccard_overlap": len(shared) / len(union),
                }
            )
    return pd.DataFrame(rows)


def package_report(
    *,
    project_root: Path,
    run_root: Path,
    evaluation_path: Path,
    report_dir: Path,
    source_train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    cache_root: Path,
    pool_root: Path,
    spectrogram_config_path: Path,
    overwrite: bool = False,
) -> Path:
    """Promote bounded low-resource evidence while leaving checkpoints under ignored runs/."""
    if report_dir.exists() and any(report_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Report directory is not empty: {report_dir}; pass --overwrite-report intentionally")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    input_audit_path = run_root / "input_audit.json"
    if not input_audit_path.is_file():
        raise FileNotFoundError(
            f"Low-resource input audit is missing: {input_audit_path}; run train-sweep first"
        )
    input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    report_dir.mkdir(parents=True, exist_ok=True)
    validation_selection = pd.DataFrame(evaluation["validation_selection"])
    test_per_run = pd.DataFrame(evaluation["test_per_run"])
    test_aggregate = pd.DataFrame(evaluation["test_aggregate"])
    paired = pd.DataFrame(evaluation["paired_deltas"])
    paired_summary = pd.DataFrame(evaluation["paired_summary"])

    source_validation = pd.read_csv(run_root / "evaluation_tables" / "validation_per_run.csv")
    source_validation.to_csv(report_dir / "validation_per_run.csv", index=False, lineterminator="\n")
    validation_selection.to_csv(report_dir / "validation_selection.csv", index=False, lineterminator="\n")
    test_per_run.to_csv(report_dir / "test_per_run.csv", index=False, lineterminator="\n")
    test_aggregate.to_csv(report_dir / "test_aggregate.csv", index=False, lineterminator="\n")
    paired.to_csv(report_dir / "paired_deltas.csv", index=False, lineterminator="\n")
    paired_summary.to_csv(report_dir / "paired_summary.csv", index=False, lineterminator="\n")

    membership_frames = []
    for subset_path in sorted((run_root / "manifests").glob("real_*_subset_*.csv")):
        frame = pd.read_csv(subset_path)
        membership_frames.append(
            frame[
                [
                    "low_resource_subset_seed",
                    "low_resource_recording_rank",
                    "name",
                    "id",
                    "relative_wav_path",
                    "audio_sha256",
                ]
            ]
        )
    if not membership_frames:
        raise FileNotFoundError(f"No low-resource subset manifests found under {run_root / 'manifests'}")
    membership = pd.concat(membership_frames, ignore_index=True)
    membership.to_csv(report_dir / "subset_membership.csv", index=False, lineterminator="\n")
    _subset_overlap_table(membership).to_csv(
        report_dir / "subset_overlap.csv", index=False, lineterminator="\n"
    )

    confusion_dir = report_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for record in evaluation["confusion_matrices"]:
        filename = (
            f"{record['condition']}_subset_{record['real_subset_seed']}_train_{record['train_seed']}.csv"
        )
        pd.DataFrame(
            record["confusion"], index=record["classes"], columns=record["classes"]
        ).to_csv(confusion_dir / filename, lineterminator="\n")

    protocol = {
        "schema_version": 1,
        **evaluation["protocol"],
        "selected_ratios": evaluation["selected_ratios"],
        "source_train_manifest": _portable(source_train_manifest, project_root),
        "validation_manifest": _portable(validation_manifest, project_root),
        "test_manifest": _portable(test_manifest, project_root),
        "real_cache": _portable(cache_root, project_root),
        "pool_root": _portable(pool_root, project_root),
        "spectrogram_config": _portable(spectrogram_config_path, project_root),
    }
    provenance = {
        "schema_version": 1,
        "git": _git_state(project_root),
        "inputs": {
            "source_train_manifest_sha256": sha256_file(source_train_manifest),
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "test_manifest_sha256": sha256_file(test_manifest),
            "real_cache_manifest_sha256": sha256_file(cache_root / "spectrogram_manifest.csv"),
            "spectrogram_config_sha256": sha256_file(spectrogram_config_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "input_audit_sha256": sha256_file(input_audit_path),
        },
        "pool_manifests": [
            {
                "model": model,
                "seed": seed,
                "path": _portable(
                    pool_reference(project_root, pool_root, model, seed).manifest,
                    project_root,
                ),
                "sha256": sha256_file(
                    pool_reference(project_root, pool_root, model, seed).manifest
                ),
            }
            for model in GENERATORS
            for seed in evaluation["protocol"]["pool_seeds"]
        ],
        "source_files": {
            "experiment_module": {
                "path": "src/bird_song/augmentation/low_resource.py",
                "sha256": sha256_file(project_root / "src/bird_song/augmentation/low_resource.py"),
            },
            "report_module": {
                "path": "src/bird_song/augmentation/low_resource_report.py",
                "sha256": sha256_file(
                    project_root / "src/bird_song/augmentation/low_resource_report.py"
                ),
            },
            "cli": {
                "path": "scripts/15_crnn_low_resource_augmentation.py",
                "sha256": sha256_file(project_root / "scripts/15_crnn_low_resource_augmentation.py"),
            },
        },
    }
    summary = {
        "schema_version": 1,
        "title": evaluation["title"],
        "selected_ratios": evaluation["selected_ratios"],
        "test_aggregate": evaluation["test_aggregate"],
        "paired_summary": evaluation["paired_summary"],
        "caveats": evaluation["caveats"],
    }
    save_json(report_dir / "protocol.json", protocol)
    save_json(report_dir / "provenance.json", provenance)
    save_json(report_dir / "summary.json", summary)
    save_json(report_dir / "input_audit.json", input_audit)
    block_count = len(evaluation["protocol"]["replicate_blocks"])
    _write_report_charts(
        report_dir,
        validation_selection,
        paired,
        block_count=block_count,
    )

    aggregate_by_condition = {
        str(row["condition"]): row for row in evaluation["test_aggregate"]
    }
    baseline = aggregate_by_condition["real_only"]
    result_lines = []
    for generator in GENERATORS:
        ratio = int(evaluation["selected_ratios"][generator])
        condition = f"{generator}_plus_{ratio}"
        aggregate = aggregate_by_condition[condition]
        paired_row = next(
            row for row in evaluation["paired_summary"] if row["generator"] == generator
        )
        result_lines.append(
            f"| {generator} + {ratio}/species | {aggregate['macro_f1_mean']:.2%} | "
            f"{paired_row['delta_macro_f1_mean']:+.2%} | "
            f"[{paired_row['delta_macro_f1_bootstrap_95_low']:+.2%}, "
            f"{paired_row['delta_macro_f1_bootstrap_95_high']:+.2%}] |"
        )
    subset_count = len(evaluation["protocol"]["real_subset_seeds"])
    train_seed_count = len(evaluation["protocol"]["train_seeds"])
    real_per_species = int(evaluation["protocol"]["real_per_species"])
    pool_seed_text = ", ".join(str(value) for value in evaluation["protocol"]["pool_seeds"])
    vae_paired = next(
        row for row in evaluation["paired_summary"] if row["generator"] == "vae_v3"
    )
    diffusion_paired = next(
        row for row in evaluation["paired_summary"] if row["generator"] == "diffusion"
    )
    subset_seed_noun = "seed" if subset_count == 1 else "seeds"
    classifier_seed_noun = "seed" if train_seed_count == 1 else "seeds"
    block_noun = "block" if block_count == 1 else "blocks"
    readme = f"""# Low-resource CRNN synthetic-augmentation evaluation

This report evaluates a simulated classifier-label-scarcity setting: the CRNN
receives {real_per_species} labeled real spectrograms per species, each from a distinct
recording ID, with optional VAE-v3 or diffusion augmentation. Both generator
families reuse the existing generation-pool seeds {pool_seed_text}; no generator was retrained
and no additional spectrogram was generated for this experiment.

All conditions use the same from-scratch CRNN architecture, optimizer-step
budget, real-only validation set, and identical post-cache masking policy for
real and generated training rows. {subset_count} real-subset {subset_seed_noun}
crossed with {train_seed_count} classifier {classifier_seed_noun} create
{block_count} matched {block_noun}.
Generator-pool seeds rotate through those blocks with a Latin square. Ratios
are selected only from validation; the real-only baseline and one selected
ratio per generator reach test.

## Held-out result

| Condition | Mean test macro F1 | Paired delta vs real-only | Descriptive block-bootstrap 95% interval |
|---|---:|---:|---:|
| Real-only, {real_per_species}/species | {baseline['macro_f1_mean']:.2%} | reference | reference |
{chr(10).join(result_lines)}

Within this design, VAE-v3 +{evaluation['selected_ratios']['vae_v3']}/species
supports a repeatable classifier-utility improvement: its macro-F1 delta was
positive in {vae_paired['delta_macro_f1_positive_blocks']}/{block_count} matched
blocks. Diffusion +{evaluation['selected_ratios']['diffusion']}/species does not
show the same stability: its delta was positive in
{diffusion_paired['delta_macro_f1_positive_blocks']}/{block_count} blocks and its
descriptive interval spans zero. Both selected ratios are the largest tested,
so the experiment identifies the best available ratio, not a saturation point
or optimum.

The interval is a deterministic bootstrap over the {block_count} matched
experiment blocks, not a test-recording confidence interval or a p-value.
Every per-block result remains in `test_per_run.csv` and `paired_deltas.csv`;
seed disagreement is not hidden by the mean. The real-data subsets overlap
because the source split has only 56 Robin, 63 Cardinal, and 95 Sparrow
recording IDs; `subset_overlap.csv` quantifies that dependence.

## Interpretation boundary

This experiment can support a claim about classifier label scarcity with access
to already-trained generators. It does not establish performance for genuinely
rare or unseen species: the evaluated classes are common in this project, the
generators were trained with more source data than the {real_per_species}
classifier-visible examples, and the project test set has prior evaluation
history. It also does not establish waveform quality or human-perceived realism.

## Package contents

- `protocol.json`: predeclared design, seed routing, selection, and test policy.
- `input_audit.json`: split isolation, subset counts, pool identities, and all
  3,600 generated-array audit results.
- `provenance.json`: exact input, pool, source-file, and evaluation identities.
- `validation_per_run.csv` and `validation_selection.csv`: all validation evidence.
- `test_per_run.csv`, `test_aggregate.csv`, and `paired_deltas.csv`: held-out evidence.
- `paired_summary.csv`: means, sample standard deviations, ranges, and intervals.
- `subset_membership.csv` and `subset_overlap.csv`: exact real-data subsets and
  their pairwise recording-ID overlap.
- `confusion_matrices/` and `figures/`: bounded diagnostic evidence.
- `SHA256SUMS.txt`: integrity manifest for the report package.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(report_dir)
    return report_dir
