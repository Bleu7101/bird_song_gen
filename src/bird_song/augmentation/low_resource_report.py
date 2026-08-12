from __future__ import annotations

import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from bird_song.augmentation.low_resource import (
    GENERATORS,
    PROTOCOL_VERSION,
    _checkpoint_evidence_contract,
    _portable,
    _training_protocol,
    experiment_conditions,
    pool_reference,
    replicate_blocks,
)
from bird_song.generation.checkpoint_models import (
    DIFFUSION_CLAMP,
    DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
    DIFFUSION_CHECKPOINT_EPOCH,
    DIFFUSION_DDIM_ETA,
    DIFFUSION_DDIM_STEPS,
    DIFFUSION_GUIDANCE,
    DIFFUSION_STORED_SAMPLER,
    DIFFUSION_TIMESTEPS,
    VAE_REPARAMETERIZATION,
    VAE_TEMPERATURE,
)
from bird_song.generation.posterior_bank_filter import (
    POSTERIOR_BANK_CONTRACT,
    POSTERIOR_BANK_EXPECTED_COUNTS,
    POSTERIOR_BANK_SOURCE_MANIFEST,
)
from bird_song.generator_safe_validation import load_generator_safe_validation_identity
from bird_song.runtime import save_json


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


def _validated_pool_provenance(
    *,
    input_audit: dict[str, Any],
    generator_evaluation_protocol: dict[str, Any],
    generator_evaluation_protocol_path: Path,
    project_root: Path,
    pool_seeds: list[int],
) -> dict[str, Any]:
    """Cross-check terminal pool identities with the report that records refresh actions."""
    if generator_evaluation_protocol.get("schema_version") != 3:
        raise ValueError("Generator-evaluation protocol must use schema version 3")
    if (
        generator_evaluation_protocol.get("models") != list(GENERATORS)
        or generator_evaluation_protocol.get("seeds") != pool_seeds
        or generator_evaluation_protocol.get("samples_per_species") != 200
    ):
        raise ValueError("Generator-evaluation protocol has the wrong model/seed/pool shape")
    contracts = generator_evaluation_protocol.get("generation_contracts")
    if not isinstance(contracts, dict):
        raise ValueError("Generator-evaluation protocol has no generation_contracts")

    vae_expected = {
        "sampling_type": "per_species_posterior_anchor_mixture",
        "temperature": VAE_TEMPERATURE,
        "reparameterization": VAE_REPARAMETERIZATION,
        "posterior_bank_contract": POSTERIOR_BANK_CONTRACT,
        "posterior_bank_source_manifest": POSTERIOR_BANK_SOURCE_MANIFEST,
        "posterior_bank_counts": POSTERIOR_BANK_EXPECTED_COUNTS,
        "posterior_bank_derivation": "filtered_existing_posterior_bank",
        "vae_checkpoint_retrained": False,
        "pool_reused_after_vae_bank_filter": False,
        "spectrograms_regenerated_after_vae_bank_filter": True,
        "generation_batch_size": 8,
    }
    diffusion_expected = {
        "sampler": "ddim",
        "timesteps": DIFFUSION_TIMESTEPS,
        "ddim_steps": DIFFUSION_DDIM_STEPS,
        "ddim_eta": DIFFUSION_DDIM_ETA,
        "guidance_weight": DIFFUSION_GUIDANCE,
        "clamp_samples": DIFFUSION_CLAMP,
        "ema_state_dict": True,
        "checkpoint_epoch": DIFFUSION_CHECKPOINT_EPOCH,
        "checkpoint_best_validation_loss": DIFFUSION_CHECKPOINT_BEST_VALIDATION_LOSS,
        "checkpoint_selection": "validation_best",
        "stored_sampler_overridden": DIFFUSION_STORED_SAMPLER,
        "pool_reused_after_vae_bank_filter": True,
        "spectrograms_regenerated_after_vae_bank_filter": False,
        "generation_batch_size": 8,
    }
    expected_by_model = {"vae_v3": vae_expected, "diffusion": diffusion_expected}
    for model, expected in expected_by_model.items():
        observed = contracts.get(model)
        mismatched = [key for key, value in expected.items() if not isinstance(observed, dict) or observed.get(key) != value]
        if mismatched:
            raise ValueError(
                f"Generator-evaluation {model} refresh contract mismatch: {mismatched}"
            )

    details = input_audit.get("pool_details")
    if not isinstance(details, list):
        raise ValueError("Low-resource input audit has no pool_details")
    expected_keys = {(model, int(seed)) for model in GENERATORS for seed in pool_seeds}
    observed_keys = [(row.get("model"), row.get("seed")) for row in details if isinstance(row, dict)]
    if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
        raise ValueError("Low-resource input audit pool model/seed matrix is incomplete or duplicated")

    promoted_models: dict[str, Any] = {}
    for model in GENERATORS:
        model_rows = sorted(
            (row for row in details if row["model"] == model),
            key=lambda row: int(row["seed"]),
        )
        expected = expected_by_model[model]
        identity_fields = [
            key
            for key in expected
            if key
            not in {
                "pool_reused_after_vae_bank_filter",
                "spectrograms_regenerated_after_vae_bank_filter",
            }
        ]
        for row in model_rows:
            generation = row.get("generation_identity", {}).get("generation", {})
            mismatched = [key for key in identity_fields if generation.get(key) != expected[key]]
            if (
                row.get("rows") != 600
                or generation.get("schema_version") != 3
                or generation.get("generator") != model
                or generation.get("seed") != row.get("seed")
                or generation.get("samples_per_class") != 200
            ):
                mismatched.append("pool_shape_or_identity")
            if mismatched:
                raise ValueError(
                    f"Low-resource input-audit pool identity mismatch for {model} "
                    f"seed {row.get('seed')}: {sorted(set(mismatched))}"
                )
        promoted_models[model] = {
            "pool_seeds": [int(row["seed"]) for row in model_rows],
            "samples_per_pool": 600,
            "generation_contract": {
                key: expected[key] for key in identity_fields
            },
            "pool_reused_after_vae_bank_filter": expected[
                "pool_reused_after_vae_bank_filter"
            ],
            "spectrograms_regenerated_after_vae_bank_filter": expected[
                "spectrograms_regenerated_after_vae_bank_filter"
            ],
        }
        if model == "vae_v3":
            promoted_models[model]["checkpoint_not_retrained"] = (
                expected["vae_checkpoint_retrained"] is False
            )
    return {
        "source_protocol": _portable(generator_evaluation_protocol_path, project_root),
        "source_schema_version": 3,
        "low_resource_pool_generation_performed": False,
        "models": promoted_models,
    }


def _validate_package_evidence(
    *,
    project_root: Path,
    run_root: Path,
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
    validation_manifest: Path,
    test_manifest: Path,
    generator_evaluation_protocol: dict[str, Any],
    generator_evaluation_protocol_path: Path,
) -> dict[str, Any]:
    protocol = evaluation.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Low-resource evaluation has no protocol object")
    if evaluation.get("protocol_version") != PROTOCOL_VERSION or protocol.get(
        "protocol_version"
    ) != PROTOCOL_VERSION:
        raise ValueError("Low-resource evaluation must expose protocol version 4 at both levels")
    if input_audit.get("schema_version") != 2 or input_audit.get(
        "protocol_version"
    ) != PROTOCOL_VERSION:
        raise ValueError("Low-resource input audit must be schema 2 / protocol 4")

    canonical_validation = load_generator_safe_validation_identity(validation_manifest, project_root)
    if generator_evaluation_protocol.get("validation_protocol") != canonical_validation:
        raise ValueError("Generator-evaluation report uses a different validation identity")
    checkpoint_contract = _checkpoint_evidence_contract(
        run_root=run_root,
        ratios=protocol["ratios_per_species"],
        real_subset_seeds=protocol["real_subset_seeds"],
        train_seeds=protocol["train_seeds"],
        pool_seeds=protocol["pool_seeds"],
    )
    expected_runs = len(
        replicate_blocks(
            protocol["real_subset_seeds"], protocol["train_seeds"], protocol["pool_seeds"]
        )
    ) * len(experiment_conditions(protocol["ratios_per_species"]))
    if (
        checkpoint_contract["checkpoint_count"] != expected_runs
        or protocol.get("checkpoint_count") != expected_runs
        or input_audit.get("expected_training_runs") != expected_runs
    ):
        raise ValueError("Evaluation, audit, and checkpoint matrix disagree on training-run count")
    expected_training_protocol = _training_protocol()
    if (
        protocol.get("training_protocol") != expected_training_protocol
        or checkpoint_contract["training_protocol"] != expected_training_protocol
    ):
        raise ValueError("Evaluation and checkpoints disagree on the protocol-4 training contract")

    portable_validation = _portable(validation_manifest, project_root)
    identity_sources = {
        "evaluation": protocol.get("validation_protocol_identity"),
        "input_audit": input_audit.get("validation_protocol_identity"),
        "checkpoints": checkpoint_contract.get("validation_protocol_identity"),
    }
    mismatched_identities = [
        label for label, identity in identity_sources.items() if identity != canonical_validation
    ]
    if mismatched_identities:
        raise ValueError(
            "Generator-safe validation identity mismatch: " + ", ".join(mismatched_identities)
        )
    for label, manifest in {
        "evaluation": protocol.get("validation_manifest"),
        "checkpoints": checkpoint_contract.get("validation_manifest"),
        "input_audit": input_audit.get("validation_protocol_identity", {}).get(
            "output_validation_manifest"
        ),
    }.items():
        if manifest != portable_validation:
            raise ValueError(f"{label} does not identify the generator-safe validation manifest")
    if protocol.get("validation_protocol") != canonical_validation["protocol"] or input_audit.get(
        "validation_protocol"
    ) != canonical_validation["protocol"]:
        raise ValueError("Evaluation/input audit generator-safe protocol path mismatch")

    validation_rows = int(canonical_validation["validation_counts"]["after"]["rows"])
    test_rows = int(canonical_validation["test_unchanged"]["rows"])
    content_rows = input_audit.get("content_safe", {}).get("rows", {})
    if content_rows.get("validation") != validation_rows or content_rows.get("test") != test_rows:
        raise ValueError("Input-audit validation/test row counts disagree with the safe boundary")
    if canonical_validation["test_unchanged"]["manifest"] != _portable(test_manifest, project_root):
        raise ValueError("Generator-safe protocol does not identify the supplied test manifest")
    if len(pd.read_csv(test_manifest)) != test_rows:
        raise ValueError("Supplied test manifest row count disagrees with the safe boundary")
    test_records = evaluation.get("test_per_run")
    if not isinstance(test_records, list) or not test_records or any(
        row.get("sample_count") != test_rows for row in test_records
    ):
        raise ValueError("Evaluation test rows do not all use the complete held-out test set")

    pool_provenance = _validated_pool_provenance(
        input_audit=input_audit,
        generator_evaluation_protocol=generator_evaluation_protocol,
        generator_evaluation_protocol_path=generator_evaluation_protocol_path,
        project_root=project_root,
        pool_seeds=[int(seed) for seed in protocol["pool_seeds"]],
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_count": expected_runs,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "validation_protocol_identity": canonical_validation,
        "pool_provenance": pool_provenance,
    }


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
    generator_evaluation_protocol: Path,
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
    generator_evaluation_protocol = generator_evaluation_protocol.resolve()
    generator_protocol = json.loads(
        generator_evaluation_protocol.read_text(encoding="utf-8")
    )
    evidence = _validate_package_evidence(
        project_root=project_root,
        run_root=run_root,
        evaluation=evaluation,
        input_audit=input_audit,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        generator_evaluation_protocol=generator_protocol,
        generator_evaluation_protocol_path=generator_evaluation_protocol,
    )
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
        "schema_version": 2,
        **evaluation["protocol"],
        "evidence_boundaries": {
            "generator_safe_validation_rows": evidence["validation_rows"],
            "held_out_test_rows": evidence["test_rows"],
        },
        "pool_provenance": evidence["pool_provenance"],
        "selected_ratios": evaluation["selected_ratios"],
        "source_train_manifest": _portable(source_train_manifest, project_root),
        "validation_manifest": _portable(validation_manifest, project_root),
        "test_manifest": _portable(test_manifest, project_root),
        "real_cache": _portable(cache_root, project_root),
        "pool_root": _portable(pool_root, project_root),
        "spectrogram_config": _portable(spectrogram_config_path, project_root),
    }
    provenance = {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "git": _git_state(project_root),
        "evidence_boundaries": {
            "checkpoint_count": evidence["checkpoint_count"],
            "generator_safe_validation_rows": evidence["validation_rows"],
            "held_out_test_rows": evidence["test_rows"],
            "validation_protocol_identity": evidence["validation_protocol_identity"],
        },
        "pool_provenance": evidence["pool_provenance"],
        "inputs": {
            "source_train_manifest": _portable(source_train_manifest, project_root),
            "validation_manifest": _portable(validation_manifest, project_root),
            "test_manifest": _portable(test_manifest, project_root),
            "real_cache_manifest": _portable(cache_root / "spectrogram_manifest.csv", project_root),
            "spectrogram_config": _portable(spectrogram_config_path, project_root),
            "evaluation": _portable(evaluation_path, project_root),
            "input_audit": _portable(input_audit_path, project_root),
            "generator_evaluation_protocol": _portable(
                generator_evaluation_protocol, project_root
            ),
        },
        "pool_manifests": [
            {
                "model": model,
                "seed": seed,
                "path": _portable(
                    pool_reference(project_root, pool_root, model, seed).manifest,
                    project_root,
                ),
            }
            for model in GENERATORS
            for seed in evaluation["protocol"]["pool_seeds"]
        ],
        "source_files": {
            "experiment_module": {
                "path": "src/bird_song/augmentation/low_resource.py",
            },
            "report_module": {
                "path": "src/bird_song/augmentation/low_resource_report.py",
            },
            "cli": {
                "path": "scripts/15_crnn_low_resource_augmentation.py",
            },
        },
    }
    summary = {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "title": evaluation["title"],
        "evidence_boundaries": {
            "generator_safe_validation_rows": evidence["validation_rows"],
            "held_out_test_rows": evidence["test_rows"],
        },
        "pool_provenance": evidence["pool_provenance"],
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
    interpretation_lines = []
    for generator in GENERATORS:
        ratio = int(evaluation["selected_ratios"][generator])
        paired_row = next(
            row for row in evaluation["paired_summary"] if row["generator"] == generator
        )
        interval_low = float(paired_row["delta_macro_f1_bootstrap_95_low"])
        interval_high = float(paired_row["delta_macro_f1_bootstrap_95_high"])
        if interval_low > 0:
            interval_position = "lies entirely above zero"
        elif interval_high < 0:
            interval_position = "lies entirely below zero"
        else:
            interval_position = "spans zero"
        display_name = "VAE-v3" if generator == "vae_v3" else "Diffusion"
        interpretation_lines.append(
            f"- **{display_name} +{ratio}/species:** mean paired macro-F1 delta "
            f"{float(paired_row['delta_macro_f1_mean']):+.2%}; positive in "
            f"{int(paired_row['delta_macro_f1_positive_blocks'])}/{block_count} matched "
            f"blocks; the descriptive interval {interval_position}."
        )
    tested_ratios = [int(value) for value in evaluation["protocol"]["ratios_per_species"]]
    selected_ratios = [int(evaluation["selected_ratios"][model]) for model in GENERATORS]
    if all(value == max(tested_ratios) for value in selected_ratios):
        selection_boundary = (
            "Both validation-selected ratios are the largest tested, so the experiment "
            "identifies the best available ratio rather than a saturation point or optimum."
        )
    else:
        selection_boundary = (
            f"Each ratio was selected only among the tested values {tested_ratios}; the result "
            "does not estimate a continuous optimum."
        )
    subset_seed_noun = "seed" if subset_count == 1 else "seeds"
    classifier_seed_noun = "seed" if train_seed_count == 1 else "seeds"
    block_noun = "block" if block_count == 1 else "blocks"
    pool_provenance = evidence["pool_provenance"]
    vae_provenance = pool_provenance["models"]["vae_v3"]
    diffusion_provenance = pool_provenance["models"]["diffusion"]
    vae_contract = vae_provenance["generation_contract"]
    bank_counts = ", ".join(
        f"{species}={count}"
        for species, count in vae_contract["posterior_bank_counts"].items()
    )
    vae_checkpoint_text = (
        "was not retrained" if vae_contract["vae_checkpoint_retrained"] is False else "was retrained"
    )
    vae_pool_text = (
        "were regenerated"
        if vae_provenance["spectrograms_regenerated_after_vae_bank_filter"]
        else "were not regenerated"
    )
    diffusion_pool_text = (
        "were reused and were not regenerated"
        if diffusion_provenance["pool_reused_after_vae_bank_filter"]
        and not diffusion_provenance["spectrograms_regenerated_after_vae_bank_filter"]
        else "do not have the expected reuse provenance"
    )
    readme = f"""# Low-resource CRNN synthetic-augmentation evaluation

This report evaluates a simulated classifier-label-scarcity setting: the CRNN
receives {real_per_species} labeled real spectrograms per species, each from a distinct
recording ID, with optional VAE-v3 or diffusion augmentation. Both generator
families consume generation-pool seeds {pool_seed_text}. The VAE checkpoint
{vae_checkpoint_text}; its filtered posterior-bank contract is
`{vae_contract['posterior_bank_contract']}` with {bank_counts}, and its three pools
{vae_pool_text}. The three diffusion pools {diffusion_pool_text}. These refresh
actions come from `{pool_provenance['source_protocol']}` and were cross-checked
against the strict input-audit identities. The low-resource workflow itself
generated no pools.

All conditions use the same from-scratch CRNN architecture, optimizer-step
budget, {evidence['validation_rows']}-row generator-safe real validation set,
{evidence['test_rows']}-row held-out real test set, and identical post-cache
masking policy for real and generated training rows. {subset_count} real-subset {subset_seed_noun}
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

{chr(10).join(interpretation_lines)}

{selection_boundary}

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
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return report_dir
