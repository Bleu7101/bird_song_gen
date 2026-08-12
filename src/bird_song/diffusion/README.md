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

## Maintained evaluation contract

The selected checkpoint remains external to `main` and is loaded from the
separate diffusion workspace. Sampling must copy the selected notebook's
inference settings exactly:

- sampler: DDIM, not DDPM
- DDIM steps: 100
- DDIM eta: 0
- classifier-free guidance weight: 3.0
- clean-sample clamp: 4 standardized units
- EMA weights: enabled

The recorded pool contract uses the validation-best epoch-34 EMA checkpoint.
The existing seed-42/123/777 pools contain 200 classifier-ready `[1,128,128]`
arrays per species under
`runs/generator_checkpoint_evaluation/pools/diffusion/`. Their metadata records
the checkpoint provenance and complete sampling contract, and all three pools
passed the current audit. They were reused for the 2026-08-12 refresh; **no
diffusion spectrogram generation was run**. Incompatible pools would still be
rejected rather than silently reused.

CRNN-only quality evidence belongs in
[`reports/generator_checkpoint_evaluation_2026-08-12`](../../../reports/generator_checkpoint_evaluation_2026-08-12/),
where Diffusion records 92.44% three-seed target-label accuracy and 92.42%
macro F1 under the selected CRNN. Copy-risk thresholds use the 510-row
generator-safe validation manifest; calibration remains on the unchanged
489-row held-out test manifest.

The completed nine-block `50+0`, `50+50`, `50+100`, and `50+200` downstream
study selected Diffusion +200/species. Mean held-out macro F1 for newly
initialized CRNNs rose from 84.75% real-only to 86.21%; the mean paired change
was +1.46 percentage points, positive in 6/9 blocks, with a descriptive
block-bootstrap interval of +0.15 to +2.89. The result is packaged in
[`reports/crnn_low_resource_augmentation_2026-08-12`](../../../reports/crnn_low_resource_augmentation_2026-08-12/)
and does not compare against the historical full-data CRNN. The generator-only
speed report is recorded in
[`reports/generator_speed_comparison_2026-08-12`](../../../reports/generator_speed_comparison_2026-08-12/):
100-step DDIM averaged 412.3090 seconds per 600 spectrograms in the retained
benchmark. Keeping inference support on `main` does not make any diffusion
implementation canonical or establish perceptual realism.
