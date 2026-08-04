from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and score blinded bird-audio listening responses.")
    parser.add_argument("--key", type=Path, default=PROJECT_ROOT / "runs/listening_study/blind_key.csv")
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/listening_study/results")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "listening_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {summary_path}; pass --overwrite")
    key = pd.read_csv(args.key)
    required_key = {"clip_id", "condition", "target_species"}
    missing_key = required_key - set(key.columns)
    if missing_key:
        raise ValueError(f"Listening key is missing columns: {sorted(missing_key)}")

    response_paths: list[Path] = []
    for value in args.responses:
        matches = [Path(path) for path in glob.glob(str(value))]
        response_paths.extend(matches or [value])
    missing_paths = [path for path in response_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Listening response file does not exist: {missing_paths[0]}")
    response_frames = []
    required_response = {
        "rater_id",
        "clip_id",
        "audio_quality",
        "bird_likeness",
        "species_choice",
    }
    for path in response_paths:
        frame = pd.read_csv(path)
        missing = required_response - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing response columns: {sorted(missing)}")
        frame["response_file"] = str(path.resolve())
        response_frames.append(frame)
    responses = pd.concat(response_frames, ignore_index=True)
    if responses[list(required_response)].isna().any().any():
        raise ValueError("Responses contain blank required fields")
    for metric in ("audio_quality", "bird_likeness"):
        responses[metric] = pd.to_numeric(responses[metric], errors="raise")
        if not responses[metric].between(1, 5).all():
            raise ValueError(f"{metric} ratings must be between 1 and 5")
    if responses.duplicated(["rater_id", "clip_id"]).any():
        raise ValueError("A rater may submit only one response per clip")
    unknown_clips = sorted(set(responses["clip_id"]) - set(key["clip_id"]))
    if unknown_clips:
        raise ValueError(f"Responses contain unknown clip IDs: {unknown_clips}")

    combined = responses.merge(key[["clip_id", "condition", "target_species"]], on="clip_id", how="left")
    combined["species_correct"] = combined["species_choice"] == combined["target_species"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "combined_listening_ratings.csv", index=False)
    per_condition = (
        combined.groupby("condition")
        .agg(
            responses=("clip_id", "size"),
            raters=("rater_id", "nunique"),
            audio_quality_median=("audio_quality", "median"),
            bird_likeness_median=("bird_likeness", "median"),
            forced_choice_species_accuracy=("species_correct", "mean"),
        )
        .reset_index()
    )
    per_condition.to_csv(args.output_dir / "listening_by_condition.csv", index=False)
    summary = {
        "raters": int(combined["rater_id"].nunique()),
        "responses": len(combined),
        "conditions": per_condition.to_dict(orient="records"),
        "gate_inputs": {
            row["condition"]: float(row["bird_likeness_median"])
            for row in per_condition.to_dict(orient="records")
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(per_condition.to_string(index=False))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
