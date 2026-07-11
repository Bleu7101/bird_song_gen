from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from bird_song.data import InferenceDataset, make_loader
from bird_song.runtime import choose_device, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".npy"}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify generated audio or generated log-mel .npy files.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="One file or a directory searched recursively.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/classifier/generated_predictions.csv")
    parser.add_argument("--labels-from-parent", action="store_true", help="Treat each file's parent folder as the expected species.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [args.input] if args.input.is_file() else sorted(p for p in args.input.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No supported audio or .npy files found under {args.input}")
    device = choose_device(args.device)
    model, classes, config, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    loader = make_loader(InferenceDataset(paths, config), args.batch_size, args.workers)
    rows = []
    with torch.inference_mode():
        for specs, batch_paths in loader:
            probabilities = model(specs.to(device, non_blocking=True)).softmax(1).cpu()
            for raw_path, probs in zip(batch_paths, probabilities):
                prediction = int(probs.argmax())
                row = {"path": raw_path, "prediction": classes[prediction], "confidence": float(probs[prediction])}
                row.update({f"p_{name}": float(probs[index]) for index, name in enumerate(classes)})
                if args.labels_from_parent:
                    expected_slug = slug(Path(raw_path).parent.name)
                    expected = next((name for name in classes if slug(name) == expected_slug), None)
                    row["expected"] = expected or Path(raw_path).parent.name
                    row["correct"] = expected == classes[prediction] if expected else None
                rows.append(row)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame[[column for column in ("path", "expected", "prediction", "confidence", "correct") if column in frame]].to_string(index=False))
    if args.labels_from_parent and frame["correct"].notna().any():
        print(f"Generated target-label rate: {frame['correct'].dropna().mean():.4f}")
    print(f"Predictions: {args.output}")


if __name__ == "__main__":
    main()
