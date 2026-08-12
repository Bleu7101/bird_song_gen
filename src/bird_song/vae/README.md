# VAE experiments on main

VAE v1, v2, and v3 are preserved on `main`. The experiment entry point is
`notebooks/04_conditional_vae.ipynb`; versioned checkpoints are under
`artifacts/models/vae/conditional_vae*/`, and curated recorded outputs are
under `reports/vae/conditional_vae*/`. Fresh notebook outputs remain under the
ignored local `outputs/` workspace.

The maintained V3 inference implementation is shared by
`src/bird_song/generation/checkpoint_models.py` and
`src/bird_song/generation/checkpoint_pool.py`. Use it with
`artifacts/models/vae/conditional_vae_v3/conditional_vae_v3_best.pt` and the
filtered existing posterior bank. The VAE checkpoint was **not retrained**.
V1 and V2 remain experiment records and are not part of the current
cross-generator study.

## Corrected V3 sampling contract

VAE temperature is applied inside reparameterization:

```text
std = exp(0.5 * logvar)
z = mu + 0.35 * std * epsilon
```

The pool metadata records both `temperature = 0.35` and the exact
reparameterization contract. An existing pool is rejected unless it has the
current metadata schema and matching contract, so samples produced before this
correction cannot be silently reused.

The existing posterior bank was filtered against the current
`content_safe_v2` training manifest. It retains 256 Northern Cardinal,
247 Song Sparrow, and 256 American Robin anchors. All three seed-42/123/777
VAE pools were then regenerated, with 200 classifier-ready `[1,128,128]`
arrays per species under
`runs/generator_checkpoint_evaluation/pools/vae_v3/`.

Their CRNN-only quality evidence belongs in
[`reports/generator_checkpoint_evaluation_2026-08-12`](../../../reports/generator_checkpoint_evaluation_2026-08-12/),
where the three-seed means are 96.67% target-label accuracy and 96.66% macro
F1. Copy-risk thresholds use the 510-row generator-safe validation manifest;
the CRNN calibration set remains the unchanged 489-row held-out test manifest.
These are classifier-view diagnostics rather than perceptual-quality scores.

The completed matched `50+0`, `50+50`, `50+100`, and `50+200` low-resource
study selected VAE-v3 +200/species. Across nine blocks, the newly initialized
CRNN mean held-out macro F1 rose from 84.75% for real-only to 87.48%; the mean
paired change was +2.72 percentage points, positive in 7/9 blocks, with a
descriptive block-bootstrap interval of +1.00 to +4.65. The result is packaged
in [`reports/crnn_low_resource_augmentation_2026-08-12`](../../../reports/crnn_low_resource_augmentation_2026-08-12/)
and does not compare against the historical full-data CRNN. Generator-only
timing is recorded separately in
[`reports/generator_speed_comparison_2026-08-12`](../../../reports/generator_speed_comparison_2026-08-12/):
VAE-v3 averaged 0.3058 seconds per 600 spectrograms after the filtered bank was
loaded.
