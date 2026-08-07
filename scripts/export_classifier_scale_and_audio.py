"""Export generated spectrograms to (1) classifier-scale .npy and (2) audio .wav.

Both outputs branch from the same generated sample (standardized scale):
  relative dB = npy * std_db + mean_db
    (1) classifier scale: normalize_logmel(relative dB) -> [-1, 1]              -> .npy
    (2) audio:            relative dB -> power -> Griffin-Lim -> waveform        -> .wav

Audio cannot be built from the classifier scale directly: Griffin-Lim needs a mel POWER
spectrogram, so the audio branch goes through relative dB / power, not [-1, 1].

Outputs (under the project's outputs/ tree):
  outputs/conditional_diffusion/generated_npy_classifier_scale/*.npy
  outputs/conditional_diffusion/generated_audio/*.wav

Run:
    python scripts/export_classifier_scale_and_audio.py
Audio export needs: pip install librosa soundfile
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Parameters.
spec_cfg = json.loads((PROJECT_ROOT / "configs" / "spectrogram.json").read_text(encoding="utf-8"))
stats = json.loads((PROJECT_ROOT / "processed" / "normalization_stats.json").read_text(encoding="utf-8"))
mean_db = float(stats.get("mean_db", stats.get("mean")))
std_db = float(stats.get("std_db", stats.get("std")))
top_db = float(spec_cfg.get("top_db", 80.0))
sr = int(spec_cfg["sample_rate"])
n_fft = int(spec_cfg["n_fft"])
hop = int(spec_cfg["hop_length"])
fmin = float(spec_cfg.get("f_min", 0.0))
fmax = float(spec_cfg.get("f_max", sr / 2))
print(f"mean_db={mean_db:.3f} std_db={std_db:.3f} top_db={top_db} | "
      f"sr={sr} n_fft={n_fft} hop={hop} fmin={fmin} fmax={fmax}")

# Input / output directories.
GEN_DIR = PROJECT_ROOT / "outputs" / "conditional_diffusion" / "generated_npy"
NPY_OUT = PROJECT_ROOT / "outputs" / "conditional_diffusion" / "generated_npy_classifier_scale"
AUDIO_OUT = PROJECT_ROOT / "outputs" / "conditional_diffusion" / "generated_audio"
NPY_OUT.mkdir(parents=True, exist_ok=True)
AUDIO_OUT.mkdir(parents=True, exist_ok=True)

# Optional audio backend.
try:
    import librosa
    import soundfile as sf

    HAVE_AUDIO = True
except ImportError as exc:  # noqa: BLE001
    HAVE_AUDIO = False
    print(f"\n[!] Audio libraries missing; skipping .wav export. "
          f"Install with: pip install librosa soundfile\n    ({exc})\n")


def standardized_to_db(x: np.ndarray) -> np.ndarray:
    """Standardized scale -> relative dB (about [-80, 0])."""
    return x * std_db + mean_db


def db_to_classifier_scale(db: np.ndarray) -> np.ndarray:
    """Relative dB -> normalize_logmel -> [-1, 1] (classifier scale)."""
    rel = np.clip(db - db.max(), -top_db, 0.0)
    return (rel * (2.0 / top_db) + 1.0).astype(np.float32)


def db_to_audio(db: np.ndarray) -> np.ndarray:
    """Relative dB -> mel power -> Griffin-Lim -> waveform (peak-normalized)."""
    mel_power = librosa.db_to_power(db.astype(np.float64))
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=n_fft, hop_length=hop, fmin=fmin, fmax=fmax, power=2.0, n_iter=64,
    )
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.97   # avoid clipping
    return audio.astype(np.float32)


files = sorted(GEN_DIR.glob("*.npy"))
if not files:
    raise SystemExit(f"No generated npy found: {GEN_DIR}")
print(f"Found {len(files)} generated npy; exporting...\n")

for index, f in enumerate(files, 1):
    x = np.load(f).astype(np.float32).squeeze()      # (128, 128) standardized scale
    db = standardized_to_db(x)

    # (1) classifier-scale npy, saved as (1, 128, 128) to match the originals.
    np.save(NPY_OUT / f.name, db_to_classifier_scale(db)[None].astype(np.float32))

    # (2) audio wav.
    tag = "npy"
    if HAVE_AUDIO:
        sf.write(AUDIO_OUT / (f.stem + ".wav"), db_to_audio(db), sr)
        tag = "npy + wav"
    print(f"  [{index:2d}/{len(files)}] {f.name}  ->  {tag}")

print("\nDone.")
print("classifier-scale npy ->", NPY_OUT)
print("audio wav            ->", AUDIO_OUT if HAVE_AUDIO else "(skipped, audio libs missing)")
print("\nCompare classifier results:")
print("  # original (standardized scale, collapses to one class):")
print("  python scripts/06_evaluate_generated.py --checkpoint "
      "classifier_artifacts/Harvey_classifier/best.pt --input outputs/conditional_diffusion/generated_npy")
print("  # converted (classifier scale, should be normal):")
print("  python scripts/06_evaluate_generated.py --checkpoint "
      "classifier_artifacts/Harvey_classifier/best.pt --input outputs/conditional_diffusion/generated_npy_classifier_scale")
