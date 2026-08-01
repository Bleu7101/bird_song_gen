# Harvey Autoregressive Transformer Model Card

## Verdict

**Overall assessment: working baseline, share with caveats.**

The transformer trained cleanly and produces valid normalized 128 x 128 log-mel spectrograms that the repository classifiers can read directly. Its validation likelihood is stable, and the published residual classifier recognizes many generated samples as the requested species. However, the images remain visibly diffuse and lack much of the crisp chirp, sweep, and harmonic structure found in real recordings. Northern Cardinal conditioning is the principal weakness. Do not present classifier agreement as proof of acoustic realism.

## Checkpoint

- File: `best.pt`
- SHA-256: `571C4E5B8CBD05F7D7209CCA0F15B5545A8828CC94AA3BC1955E8BA8B20011F0`
- Model type: species-conditional autoregressive spectrogram transformer
- Trainable parameters: 4,953,856
- Selected epoch: 55 of 60
- Best validation NLL: -0.629258
- Final validation NLL: -0.628472
- Training data: 2,339 cached spectrograms
- Validation data: 519 cached spectrograms
- Hardware: NVIDIA GeForce RTX 4070 SUPER
- Total recorded epoch time: 68.7 seconds

The final training loss was -0.574484. Validation remaining slightly better than training is consistent with dropout being active during training and disabled for validation; the curve does not show obvious overfitting or numerical instability.

## Classifier compatibility control

The published residual classifier was evaluated through the same `.npy` loading path on 489 real cached held-out spectrograms. It reproduced its recorded test accuracy exactly: **442/489 = 90.39%**, with 86.93% mean confidence. This confirms that generated `.npy` files, resizing, normalization, and classifier input shape are wired correctly.

## Generated-sample evaluation

Each temperature used 64 newly generated samples per species with generation seed 2026. The metric below is forced-choice classifier agreement with the requested species, not perceptual realism.

| Temperature | Target-label agreement | Mean confidence |
|---:|---:|---:|
| 0.4 | 33.85% | 60.83% |
| 0.6 | 55.73% | 64.45% |
| 0.8 | 66.15% | 74.16% |
| **1.0** | **80.73%** | **74.87%** |
| 1.2 | 75.52% | 69.47% |

At temperature 1.0, the published residual classifier agreed with the target on 155 of 192 samples:

| Intended species | Agreement | Correct / total |
|---|---:|---:|
| American Robin | 95.31% | 61 / 64 |
| Northern Cardinal | 48.44% | 31 / 64 |
| Song Sparrow | 98.44% | 63 / 64 |

Temperature changes the class bias rather than improving all species uniformly. At 1.2, Cardinal agreement increased to 95.31%, but Robin fell to 50.00% and Sparrow to 81.25%. The newer CRNN classifier agreed with only 39.58% of temperature-1.0 samples and predicted Robin for 180 of 192 images. That cross-classifier disagreement is strong evidence that the samples remain outside the real-audio training distribution.

## Visual review

- [Generated temperature-1.0 preview](../../outputs/autoregressive_transformer_preview_t10/conditional_samples.png)
- [Real held-out comparison from different recordings](../../outputs/autoregressive_transformer_preview_t10/real_test_samples.png)

The generated preview contains broad horizontal energy bands and noisy texture but few sharply localized notes or harmonic trajectories. Real test spectrograms show substantially clearer time-frequency structure and more within-species variety.

## Recommended use

- Use `temperature=1.0` as the best tested global sampling setting.
- Use classifier target agreement only as one diagnostic alongside listening tests, real-versus-generated spectrogram review, and diversity checks.
- Treat the checkpoint as a Stage 6 baseline for further generator work, not as a realism-ready synthesis model.
- For the next model iteration, prioritize a finer or latent representation. The current 16 x 16 continuous patches with Gaussian likelihood encourage smooth averages; smaller patches, learned discrete/latent tokens, or a latent diffusion decoder should preserve sharper bird-note structure.

## Reproduce generation and scoring

```powershell
python scripts/06_generate_transformer.py `
  --checkpoint runs/transformer_generator/best.pt `
  --output-dir outputs/autoregressive_transformer_eval_t10 `
  --samples-per-species 64 --temperature 1.0 --seed 2026 --device cuda --no-figure

python scripts/07_evaluate_generated.py `
  --checkpoint classifier_artifacts/Harvey_classifier/best.pt `
  --input outputs/autoregressive_transformer_eval_t10 `
  --labels-from-parent `
  --output runs/transformer_generator/classifier_eval_residual_t10.csv `
  --batch-size 128 --workers 0 --device cuda
```

## Included evidence

- `history.csv`: all 60 training epochs.
- `config.json`: portable training configuration.
- `classifier_eval_real_cached_test.*`: `.npy` compatibility control.
- `classifier_eval_residual_*`: published residual-classifier temperature sweep.
- `classifier_eval_crnn_*`: secondary cross-classifier check.
- `../../outputs/autoregressive_transformer_eval*`: generated evaluation arrays and manifests.
- `../../outputs/autoregressive_transformer_preview*`: preview arrays and comparison figures.
