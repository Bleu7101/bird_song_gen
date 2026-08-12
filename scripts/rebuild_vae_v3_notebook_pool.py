from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The local Python 3.14 environment does not have pywin32.  Jupyter's
# Windows ACL fallback then rejects its connection file before the kernel can
# start. Explicitly opt into Jupyter's documented fallback before importing
# nbclient (the flag is read at import time); the connection file stays in
# Jupyter's runtime directory and is not part of the frozen pool.
os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "1")

import nbformat
import pandas as pd
from nbclient import NotebookClient

from bird_song.evaluation.provenance import git_revision, sha256_file
from bird_song.evaluation.vae_v3_pool import (
    build_array_hashes,
    build_generation_notebook,
    build_pool_metadata,
    ensure_fresh_output,
    make_portable_manifest,
)
from bird_song.runtime import choose_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild a frozen VAE-v3 evaluation pool from notebook 04."
    )
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--posterior-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-species", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in (
        ("notebook", args.notebook),
        ("project root", args.project_root),
        ("checkpoint", args.checkpoint),
        ("posterior bank", args.posterior_bank),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if not args.notebook.is_file() or args.notebook.suffix.lower() != ".ipynb":
        raise ValueError(f"notebook must be an .ipynb file: {args.notebook}")
    if not args.checkpoint.is_file() or not args.posterior_bank.is_file():
        raise ValueError("checkpoint and posterior bank must be files")
    if args.samples_per_species < 1:
        raise ValueError("samples-per-species must be positive")
    if args.seed < 0 or args.temperature < 0:
        raise ValueError("seed must be non-negative and temperature cannot be negative")


def freeze_outputs(args: argparse.Namespace, executed_notebook, *, device_used: str) -> None:
    output = args.output.resolve()
    raw_manifest_path = output / "generated_manifest.csv"
    if not raw_manifest_path.is_file():
        raise FileNotFoundError(f"notebook did not write its generated manifest: {raw_manifest_path}")

    raw_manifest = pd.read_csv(raw_manifest_path)
    expected_count = args.samples_per_species * 3
    if len(raw_manifest) != expected_count:
        raise RuntimeError(
            f"notebook generated {len(raw_manifest)} rows; expected {expected_count}"
        )
    counts = raw_manifest["name"].value_counts()
    if counts.empty or not (counts == args.samples_per_species).all():
        raise RuntimeError(f"unexpected per-species counts: {counts.to_dict()}")

    portable = make_portable_manifest(raw_manifest, output)
    portable.to_csv(output / "manifest.csv", index=False)
    hashes = build_array_hashes(portable, output)
    hashes.to_csv(output / "array_hashes.csv", index=False)

    label_to_id = {
        str(name): int(label)
        for name, label in portable[["name", "label_id"]]
        .drop_duplicates()
        .sort_values("label_id")
        .itertuples(index=False, name=None)
    }
    metadata = build_pool_metadata(
        notebook=args.notebook,
        project_root=args.project_root,
        checkpoint=args.checkpoint,
        posterior_bank=args.posterior_bank,
        output=output,
        notebook_sha256=sha256_file(args.notebook),
        checkpoint_sha256=sha256_file(args.checkpoint),
        posterior_bank_sha256=sha256_file(args.posterior_bank),
        git_revision=git_revision(PROJECT_ROOT),
        seed=args.seed,
        generation_stream_seed=args.seed + 100,
        samples_per_species=args.samples_per_species,
        temperature=args.temperature,
        device_used=device_used,
        label_to_id=label_to_id,
    )
    metadata["files"] = {
        "manifest": "manifest.csv",
        "raw_notebook_manifest": "generated_manifest.csv",
        "array_hashes": "array_hashes.csv",
        "executed_notebook": "executed_04_conditional_vae_pool.ipynb",
    }
    (output / "generation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    nbformat.write(executed_notebook, output / "executed_04_conditional_vae_pool.ipynb")


def main() -> None:
    args = build_parser().parse_args()
    args.notebook = args.notebook.resolve()
    args.project_root = args.project_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.posterior_bank = args.posterior_bank.resolve()
    args.output = args.output.resolve()
    validate_inputs(args)
    ensure_fresh_output(args.output)

    device_used = str(choose_device(args.device))
    source = nbformat.read(args.notebook, as_version=4)
    prepared = build_generation_notebook(
        source,
        project_root=args.project_root,
        checkpoint=args.checkpoint,
        posterior_bank=args.posterior_bank,
        output=args.output,
        samples_per_species=args.samples_per_species,
        seed=args.seed,
        temperature=args.temperature,
        device=device_used,
    )
    client = NotebookClient(
        prepared,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(args.project_root)}},
    )
    executed = client.execute()
    freeze_outputs(args, executed, device_used=device_used)
    print(f"pool={args.output / 'manifest.csv'}")
    print(f"metadata={args.output / 'generation.json'}")


if __name__ == "__main__":
    main()
