from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIES = {
    "american_robin": "American Robin",
    "northern_cardinal": "Northern Cardinal",
    "song_sparrow": "Song Sparrow",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_mapping(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} must use NAME=VALUE syntax")
    name, item = value.split("=", 1)
    if not name.strip() or not item.strip():
        raise argparse.ArgumentTypeError(f"{option} needs non-empty NAME and VALUE")
    return name.strip(), item.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a randomized, blinded, balanced listening-study pack.")
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME=DIRECTORY",
        help="Repeat for original, bigvgan, vae_reconstruction, and vae_generated.",
    )
    parser.add_argument(
        "--species-label",
        action="append",
        default=[],
        metavar="DIRECTORY_NAME=DISPLAY_NAME",
    )
    parser.add_argument("--clips-per-species-condition", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/listening_study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clips_per_species_condition < 1:
        raise ValueError("clips-per-species-condition must be positive")
    conditions = dict(parse_mapping(value, "--condition") for value in args.condition)
    if len(conditions) != len(args.condition):
        raise ValueError("Condition names must be unique")
    species = dict(DEFAULT_SPECIES)
    species.update(dict(parse_mapping(value, "--species-label") for value in args.species_label))
    key_path = args.output_dir / "blind_key.csv"
    if key_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {key_path}; pass --overwrite")

    candidates = []
    expected_species = set(species)
    for condition, directory_value in conditions.items():
        directory = Path(directory_value).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Condition directory does not exist: {directory}")
        condition_species = set()
        for path in sorted(directory.rglob("*.wav")):
            species_slug = slug(path.parent.name)
            if species_slug in species:
                condition_species.add(species_slug)
                candidates.append(
                    {
                        "condition": condition,
                        "species_slug": species_slug,
                        "target_species": species[species_slug],
                        "source_path": path.resolve(),
                    }
                )
        missing = expected_species - condition_species
        if missing:
            raise ValueError(f"Condition {condition!r} has no WAV files for: {sorted(missing)}")

    frame = pd.DataFrame(candidates)
    chosen = []
    for condition_index, condition in enumerate(conditions):
        for species_index, species_slug in enumerate(species):
            subset = frame.loc[
                (frame["condition"] == condition) & (frame["species_slug"] == species_slug)
            ]
            count = args.clips_per_species_condition
            if len(subset) < count:
                raise ValueError(
                    f"Condition {condition!r}, species {species_slug!r} has {len(subset)} clips; need {count}"
                )
            chosen.append(
                subset.sample(
                    count,
                    random_state=args.seed + condition_index * 100 + species_index,
                )
            )
    selected = pd.concat(chosen).reset_index(drop=True)
    order = list(range(len(selected)))
    random.Random(args.seed).shuffle(order)
    selected = selected.iloc[order].reset_index(drop=True)

    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for index, row in selected.iterrows():
        clip_id = f"clip_{index + 1:03d}"
        destination = clips_dir / f"{clip_id}.wav"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        shutil.copy2(row["source_path"], destination)
        key_rows.append(
            {
                "clip_id": clip_id,
                "condition": row["condition"],
                "target_species": row["target_species"],
                "source_path": str(row["source_path"]),
                "blind_path": str(destination.resolve()),
            }
        )

    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    response = pd.DataFrame(
        {
            "rater_id": [""] * len(key_rows),
            "clip_id": [row["clip_id"] for row in key_rows],
            "audio_quality": [""] * len(key_rows),
            "bird_likeness": [""] * len(key_rows),
            "species_choice": [""] * len(key_rows),
            "notes": [""] * len(key_rows),
        }
    )
    response.to_csv(args.output_dir / "response_template.csv", index=False)
    instructions = (
        "Rate each randomized clip without opening blind_key.csv. Use integer scores from 1 (worst) "
        "to 5 (best) for audio_quality and bird_likeness. For species_choice, enter exactly one of: "
        f"{', '.join(species.values())}. Each listener should copy response_template.csv, fill every row, "
        "and use a unique rater_id. Combine and score responses only after collection.\n"
    )
    (args.output_dir / "INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
    print(f"Prepared {len(key_rows)} blinded clips across {len(conditions)} conditions.")
    print(f"Participant template: {args.output_dir / 'response_template.csv'}")
    print(f"Keep the condition key private until ratings are complete: {key_path}")


if __name__ == "__main__":
    main()
