from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen legacy residual CNN and CRNN on generated BigVGAN WAVs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, default=PROJECT_ROOT.parent / "bird_song_gen")
    parser.add_argument(
        "--residual-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--crnn-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    legacy_root = args.legacy_root.resolve()
    input_dir = args.input.resolve()
    residual = args.residual_checkpoint or legacy_root / "classifier_artifacts/Harvey_classifier/best.pt"
    crnn = args.crnn_checkpoint or legacy_root / "classifier_artifacts/selected_crnn/best.pt"
    evaluator = legacy_root / "scripts/07_evaluate_generated.py"
    for path in (input_dir, residual, crnn, evaluator):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = sorted(input_dir.rglob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"No WAV files found under {input_dir}")
    staging_dir = output_dir / ".wav_only"
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite staging directory {staging_dir}")
    for source in audio_paths:
        destination = staging_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(legacy_root / "src")
    summaries: dict[str, object] = {
        "input": str(input_dir),
        "legacy_root": str(legacy_root),
        "classifiers": {},
        "note": "Classifier agreement is a conditioning diagnostic, not an acoustic realism score.",
    }
    try:
        for name, checkpoint in (("residual_cnn", residual), ("crnn", crnn)):
            output = output_dir / f"{name}.csv"
            command = [
                sys.executable,
                str(evaluator),
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(staging_dir),
                "--output",
                str(output),
                "--labels-from-parent",
                "--workers",
                str(args.workers),
                "--device",
                args.device,
            ]
            subprocess.run(command, cwd=legacy_root, env=environment, check=True)
            summary_path = output.with_suffix(".summary.json")
            summaries["classifiers"][name] = json.loads(summary_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"Saved classifier diagnostics to {summary_path}")


if __name__ == "__main__":
    main()
