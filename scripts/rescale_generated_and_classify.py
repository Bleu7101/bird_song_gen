"""Rescale generated spectrograms from the generative scale to the classifier scale,
then classify them -- with a real-data control.

Why this exists: generated/processed .npy are in notebook-02 STANDARDIZED units
(mean 0, std 1), but the classifier expects normalize_logmel output in [-1, 1].
Feeding standardized .npy straight into scripts/06_evaluate_generated.py mis-normalizes
them (squashes to ~[0.87, 1.0]) and collapses every sample to a single class. This
script undoes the standardization back to dB, re-applies the classifier's
normalize_logmel, and classifies -- the fair comparison.

The same conversion is applied to a sample of real test spectrograms as a control:
  - real ~90% but generated collapses  -> a model/generation problem (normalization ruled out)
  - both collapse                       -> the conversion is still off

Run (paths are resolved from this file's location; reads only, writes nothing):
    python scripts/rescale_generated_and_classify.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bird_song.runtime import choose_device, load_checkpoint  # noqa: E402


device = choose_device("auto")

# Load the trained classifier.
ckpt_path = PROJECT_ROOT / "classifier_artifacts" / "Harvey_classifier" / "best.pt"
model, classes, config, _ = load_checkpoint(ckpt_path, device)
model.eval()
classes = list(classes)
top_db = float(config.top_db)

# Standardization stats used by the generative pipeline (notebook 02).
stats = json.loads((PROJECT_ROOT / "processed" / "normalization_stats.json").read_text(encoding="utf-8"))
mean_db = float(stats.get("mean_db", stats.get("mean")))
std_db = float(stats.get("std_db", stats.get("std")))
print(f"classes = {classes}")
print(f"mean_db = {mean_db:.3f} | std_db = {std_db:.3f} | top_db = {top_db}")


def generative_scale_to_classifier_scale(arr) -> torch.Tensor:
    """Standardized (mean 0, std 1) -> relative dB -> normalize_logmel [-1, 1]."""
    x = torch.as_tensor(np.asarray(arr), dtype=torch.float32).squeeze()
    db = x * std_db + mean_db                             # undo standardization -> dB
    rel = (db - db.amax()).clamp(min=-top_db, max=0.0)    # per-clip max, clamp (as normalize_logmel)
    spec = rel * (2.0 / top_db) + 1.0                     # -> [-1, 1]
    return spec.reshape(1, 1, *spec.shape).to(device)


@torch.no_grad()
def predict(arr):
    probs = model(generative_scale_to_classifier_scale(arr)).softmax(1)[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


slug_to_class = {slug(name): name for name in classes}


def report(title, items):
    counts, confs, correct, total = Counter(), [], 0, 0
    for intended, arr in items:
        pred, conf = predict(arr)
        counts[pred] += 1
        confs.append(conf)
        if intended is not None:
            total += 1
            correct += int(pred == intended)
    print(f"\n=== {title}  (n={len(items)}) ===")
    if not items:
        print("  (no samples -- check paths)")
        return
    print("  predicted counts :", dict(counts))
    print(f"  mean confidence  : {sum(confs) / len(confs):.3f}  (chance = 0.333)")
    if total:
        print(f"  target accuracy  : {correct}/{total} = {correct / total:.1%}")


# (1) Generated samples.
gen_dir = PROJECT_ROOT / "outputs" / "conditional_diffusion" / "generated_npy"
gen_items = []
for f in sorted(gen_dir.glob("*.npy")):
    match = re.match(r"diff_(.+)_\d+\.npy", f.name)
    intended = slug_to_class.get(match.group(1)) if match else None
    gen_items.append((intended, np.load(f)))
report("GENERATED (after scale conversion)", gen_items)

# (2) Real test control (identical conversion).
try:
    import pandas as pd

    manifest = PROJECT_ROOT / "processed" / "manifests" / "logmel_128_test.csv"
    frame = pd.read_csv(manifest)
    name_col = next(c for c in ("name", "species", "species_name", "label_name") if c in frame.columns)
    path_col = next(
        c for c in ("relative_spec_path", "spec_path", "spectrogram_path",
                    "logmel_path", "npy_path", "processed_path")
        if c in frame.columns
    )
    frame = frame[frame[name_col].isin(classes)]
    sample = frame.sample(min(90, len(frame)), random_state=0)

    def resolve(value):
        raw = Path(str(value))
        candidates = (
            [raw]
            if raw.is_absolute()
            else [PROJECT_ROOT / raw, PROJECT_ROOT / "processed" / raw, manifest.parent / raw]
        )
        return next((c for c in candidates if c.exists()), None)

    real_items = []
    for _, row in sample.iterrows():
        path = resolve(row[path_col])
        if path is not None:
            real_items.append((row[name_col], np.load(path)))
    report("REAL test control (same conversion, expect ~90%)", real_items)
except Exception as exc:  # noqa: BLE001
    print("\nReal control skipped:", repr(exc))

print("\nInterpretation:")
print("  - real ~90% but generated collapses -> model/generation issue (normalization ruled out)")
print("  - both collapse                     -> conversion still off")
