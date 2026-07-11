from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from bird_song.data import ManifestDataset, make_loader, resolve_dataset_root
from bird_song.runtime import choose_device, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on a labeled manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests/full_dataset_test.csv")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/classifier/test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, classes, config, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = ManifestDataset(args.manifest, resolve_dataset_root(PROJECT_ROOT, args.dataset_root), classes, config)
    loader = make_loader(dataset, args.batch_size, args.workers)
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    paths: list[str] = []
    with torch.inference_mode():
        for specs, labels, batch_paths in loader:
            probs = model(specs.to(device, non_blocking=True)).softmax(1).cpu()
            predictions.extend(probs.argmax(1).tolist())
            targets.extend(labels.tolist())
            probabilities.extend(probs.numpy())
            paths.extend(batch_paths)

    report = classification_report(targets, predictions, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(targets, predictions, labels=range(len(classes)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"checkpoint_epoch": checkpoint.get("epoch"), "report": report}, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    for path, target, prediction, probs in zip(paths, targets, predictions, probabilities):
        row = {"path": path, "target": classes[target], "prediction": classes[prediction], "confidence": float(probs[prediction])}
        row.update({f"p_{name}": float(probs[index]) for index, name in enumerate(classes)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(args.output_dir / "confusion_matrix.csv")

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="True")
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    print(classification_report(targets, predictions, target_names=classes, zero_division=0))
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
