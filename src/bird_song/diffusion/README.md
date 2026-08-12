# Diffusion branch guide

`main` does not select or package a canonical diffusion model. Diffusion-model
implementations and experiment artifacts are maintained on the separate
`Diffusion`, `Difussion`, and `diffusion_vincent` branches.

This directory is only a namespace placeholder on `main`; it should not be
interpreted as the implementation for those branches. Use the README,
notebooks, source, configs, checkpoints, and recorded evidence from the
specific diffusion branch being reviewed. Before comparing a diffusion result
with a `main` generator, verify that preprocessing, class order, dataset split,
checkpoint format, and decoder path match.

## Frozen pool used by the first augmentation evaluation

`main` keeps one local, ignored classifier-ready cache at
`artifacts/generated_spectrograms/diffusion/` solely so the first CRNN
augmentation pool can be reused by maintained experiments without generating
duplicate arrays. It contains 200 `[1,128,128]` arrays per species from the
recorded `diffusion_vincent` EMA checkpoint.
The pool used deterministic 100-step DDIM sampling, eta 0, guidance weight 3.0,
and clean-sample clamping at 4 standardized units. The checkpoint itself is
external and is not packaged on `main`.

Validation selected 200 generated samples per species. Across CRNN seeds 42,
123, and 777, that arm reached 89.23% mean held-out accuracy and 89.28% mean
macro F1, respectively 0.75 and 0.88 percentage points below the historical
selected CRNN. This is no demonstrated mean gain, not evidence that diffusion
data is harmful: the reference is one historical WAV-trained seed, not a
matched three-seed cached-real-only arm, and seed variation was material.

The complete result and its exact-content split audit are in
[`reports/crnn_synthetic_augmentation_2026-08-09`](../../../reports/crnn_synthetic_augmentation_2026-08-09/README.md).
Caching this frozen output pool and publishing its classifier-utility result do
not make the diffusion implementation canonical on `main` and do not establish
perceptual realism.
