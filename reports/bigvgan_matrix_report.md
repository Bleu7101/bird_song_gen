# BigVGAN generator retraining evidence

This report was generated from the worktree now tracked by `BigVGAN_decode` (originally named `decoder_test`). It is the first recorded baseline for future decoder/vocoder comparisons. Checkpoints are selected from validation evidence; decoded audio and frozen-classifier results are separate diagnostics.

## Contract

- BigVGAN: `nvidia/bigvgan_v2_22khz_80band_256x`; 22050 Hz; 65536 samples; raw log-mels `80×256`.
- Scaler: `{'count': 47902720, 'fitted_split': 'train', 'maximum': 2.1087865829467773, 'minimum': -11.512925148010254, 'normalization': 'global_train_minmax_to_minus_one_one'}`.
- Classes: American Robin, Northern Cardinal, Song Sparrow.

## Validation-only matrix

| Run | Seed | Best epoch | Validation metric |
|---|---:|---:|---:|
| `transformer_16x16_seed123` | 123 | 46 | -1.038747 |
| `transformer_16x16_seed42` | 42 | 49 | -1.041602 |
| `transformer_16x16_seed777` | 777 | 42 | -1.039412 |
| `transformer_8x16_seed123` | 123 | 52 | -1.143977 |
| `transformer_8x16_seed42` | 42 | 51 | -1.146163 |
| `transformer_8x16_seed777` | 777 | 48 | -1.143821 |
| `wgan_current_seed123` | 123 | 20 | 0.063891 |
| `wgan_current_seed42` | 42 | 14 | 0.198135 |
| `wgan_current_seed777` | 777 | 19 | 0.118256 |
| `wgan_stability_seed123` | 123 | 18 | 0.164725 |
| `wgan_stability_seed42` | 42 | 18 | 0.044130 |
| `wgan_stability_seed777` | 777 | 18 | 0.077806 |

WGAN selection: `stability` (mean error 0.095554, median 0.077806); selected seed 42.
Transformer selection: `8x16` (mean validation NLL -1.144653, median -1.143977); selected seed 42.

## Output gates

All recorded balanced sets below have finite 80×256 mels, valid 65,536-sample waveforms, zero silent samples, and clipped fraction below 0.001. The temperature sweep is reported for listening choice; it is not a validation-loss selection criterion.

| Set | Status | Samples | Temperature | Max clipping | Saturation | Diversity |
|---|---|---:|---:|---:|---:|---:|
| `runs/final_evaluations/transformer_8x16_seed123/temp_0.8` | generation_pass | 24 | 0.8 | 0.000000 | 0.000049 | 21.919 |
| `runs/final_evaluations/transformer_8x16_seed42/temp_0.8` | generation_pass | 24 | 0.8 | 0.000000 | 0.000051 | 21.673 |
| `runs/final_evaluations/transformer_8x16_seed777/temp_0.8` | generation_pass | 24 | 0.8 | 0.000000 | 0.000047 | 20.161 |
| `runs/final_evaluations/transformer_selected/temp_0.4` | generation_pass | 24 | 0.4 | 0.000000 | 0.000000 | 14.213 |
| `runs/final_evaluations/transformer_selected/temp_0.6` | generation_pass | 24 | 0.6 | 0.000000 | 0.000008 | 17.510 |
| `runs/final_evaluations/transformer_selected/temp_0.8` | generation_pass | 24 | 0.8 | 0.000000 | 0.000051 | 21.673 |
| `runs/final_evaluations/transformer_selected/temp_1.0` | generation_pass | 24 | 1.0 | 0.000137 | 0.000193 | 26.768 |
| `runs/final_evaluations/wgan_current_seed123` | integration_pass | 24 |  | 0.000076 | 0.000000 | 45.876 |
| `runs/final_evaluations/wgan_current_seed42` | integration_pass | 24 |  | 0.000092 | 0.000000 | 49.490 |
| `runs/final_evaluations/wgan_current_seed777` | integration_pass | 24 |  | 0.000183 | 0.000000 | 37.369 |
| `runs/final_evaluations/wgan_stability_seed123` | integration_pass | 24 |  | 0.000229 | 0.000000 | 37.457 |
| `runs/final_evaluations/wgan_stability_seed42` | integration_pass | 24 |  | 0.000290 | 0.000000 | 48.655 |
| `runs/final_evaluations/wgan_stability_seed777` | integration_pass | 24 |  | 0.000397 | 0.000000 | 46.058 |

## Conditioning diagnostics

The frozen legacy residual CNN and CRNN were run on the generated BigVGAN WAVs only. Their agreement is a conditioning diagnostic, not a perceptual realism score and is not mixed into validation selection.

| Diagnostic set | Residual CNN accuracy | CRNN accuracy |
|---|---:|---:|
| `reports/legacy_classifier_diagnostics/transformer_8x16_seed123_temp_0.8` | 0.708 | 0.417 |
| `reports/legacy_classifier_diagnostics/transformer_8x16_seed42_temp_0.8` | 0.792 | 0.458 |
| `reports/legacy_classifier_diagnostics/transformer_8x16_seed777_temp_0.8` | 0.833 | 0.500 |
| `reports/legacy_classifier_diagnostics/wgan_stability_seed123` | 0.917 | 0.875 |
| `reports/legacy_classifier_diagnostics/wgan_stability_seed42` | 0.917 | 0.958 |
| `reports/legacy_classifier_diagnostics/wgan_stability_seed777` | 0.958 | 0.792 |

## Selected artifacts

The publication bundle contains only the selected generator checkpoints, configs, hashes, model cards, and a small balanced WAV set. Downloaded BigVGAN weights, resumable checkpoints, caches, bulk arrays, and bulk audio remain excluded.

- `reports/canonical_models/transformer_8x16_seed42.pt` — SHA-256 `e9ac12cb8b9f4a878af4f5925789da24bbd6649a9a6f181114968601924bb84e` (19547414 bytes).
- `reports/canonical_models/wgan_gp_stability_seed42.pt` — SHA-256 `c760ad2d396f7a307d4f7db66267e22cd3e50e92c9fd608b39c3b4d6dbf67811` (41076610 bytes).

## Caveat

Automatic gates establish a compatible and finite decoder handoff. They do not establish perceptual realism; listen to the curated WAVs and treat the classifier outputs as diagnostics only.
