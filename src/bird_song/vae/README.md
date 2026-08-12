# VAE experiments on main

VAE v1, v2, and v3 are preserved on `main`. The experiment entry point is
`notebooks/04_conditional_vae.ipynb`, which is intentionally retained unchanged
as the historical experiment record. Versioned checkpoints are under
`artifacts/models/vae/conditional_vae*/`. Curated recorded outputs are under
`reports/vae/conditional_vae*/`; fresh notebook outputs remain under the
ignored local `outputs/` workspace.

This package directory remains a namespace placeholder rather than a second
VAE implementation. Use the notebook and its recorded artifacts together so
the architecture version, preprocessing, checkpoint path, and evaluation
evidence stay aligned. A separate VAE branch is no longer required because all
three recorded versions are retained on `main`.

## V3 augmentation evaluation

The first CRNN synthetic-augmentation evaluation used exactly one reusable V3
pool at `artifacts/generated_spectrograms/vae_v3/`: 200 classifier-ready
`[1,128,128]` arrays per species, generated with seed 42 and posterior-bank
temperature 0.35. The source checkpoint is
`artifacts/models/vae/conditional_vae_v3/conditional_vae_v3_best.pt`.
V1 and V2 were not evaluated in this sweep.

Validation selected 200 generated samples per species. Across CRNN seeds 42,
123, and 777, that arm reached 89.43% mean held-out accuracy and 89.48% mean
macro F1, respectively 0.55 and 0.68 percentage points below the historical
selected CRNN. This is no demonstrated mean gain, not evidence that VAE data is
harmful: the comparator is one historical WAV-trained seed rather than a
matched three-seed cached-real-only arm, and seed variation was material.

See
[`reports/crnn_synthetic_augmentation_2026-08-09`](../../../reports/crnn_synthetic_augmentation_2026-08-09/README.md)
for the full protocol, per-seed results, provenance, cache audit, and
interpretation limits. This classifier-utility result is not a perceptual
realism or waveform-quality claim.
