# Harvey Autoregressive Transformer Model Card

## Verdict

**Overall assessment: working baseline, share with caveats.**

The transformer trained cleanly and produces valid normalized 128 x 128 log-mel
spectrograms that the selected CRNN can read directly. Its validation likelihood
is stable, but CRNN target-label agreement is weak and the images remain visibly
diffuse, without much of the crisp chirp, sweep, and harmonic structure found in
real recordings. Northern Cardinal and Song Sparrow conditioning are the main
weaknesses. Do not present classifier agreement as proof of acoustic realism.

## Checkpoint

- File: `best.pt`
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

## Generated-sample evaluation

Each retained setting used 64 newly generated samples per species with generation
seed 2026. The metric below is forced-choice agreement from the selected CRNN,
not perceptual realism.

| Temperature | Target-label agreement | Mean confidence |
|---:|---:|---:|
| 0.8 | 37.50% | 66.25% |
| **1.0** | **39.58%** | **73.04%** |

At temperature 1.0, the selected CRNN agreed with the target on 76 of 192
samples and predicted American Robin for 180 of them:

| Intended species | Agreement | Correct / total |
|---|---:|---:|
| American Robin | 100.00% | 64 / 64 |
| Northern Cardinal | 9.38% | 6 / 64 |
| Song Sparrow | 9.38% | 6 / 64 |

The two retained CRNN-scored settings show the same severe Robin bias. The
temperature-1.0 result is slightly stronger overall, but neither setting
supports reliable three-species conditioning.

## Visual review

- [Generated temperature-1.0 preview](figures/temperature_1.0.png)
- [Real held-out comparison from different recordings](figures/real_test_samples.png)

The generated preview contains broad horizontal energy bands and noisy texture but few sharply localized notes or harmonic trajectories. Real test spectrograms show substantially clearer time-frequency structure and more within-species variety.

## Recommended use

- Use `temperature=1.0` as the stronger of the two retained CRNN-scored settings.
- Use classifier target agreement only as one diagnostic alongside listening tests, real-versus-generated spectrogram review, and diversity checks.
- Treat the checkpoint as a Stage 6 baseline for further generator work, not as a realism-ready synthesis model.
- For the next model iteration, prioritize a finer or latent representation. The current 16 x 16 continuous patches with Gaussian likelihood encourage smooth averages; smaller patches, learned discrete/latent tokens, or a latent diffusion decoder should preserve sharper bird-note structure.

## Reproduce generation and scoring

The command below reproduces the retained CRNN temperature-1.0 score.

```powershell
python scripts/06_generate_transformer.py `
  --checkpoint runs/transformer_generator/best.pt `
  --output-dir outputs/autoregressive_transformer_eval_t10 `
  --samples-per-species 64 --temperature 1.0 --seed 2026 --device cuda --no-figure

python scripts/07_evaluate_generated.py `
  --checkpoint artifacts/models/classifier/selected_crnn/best.pt `
  --input outputs/autoregressive_transformer_eval_t10 `
  --labels-from-parent `
  --output runs/transformer_generator/classifier_eval_crnn_t10.csv `
  --batch-size 128 --workers 0 --device cuda
```

## Included evidence

- `history.csv`: all 60 training epochs.
- `config.json`: portable training configuration.
- `classifier_eval_crnn_*`: selected-CRNN generated-sample diagnostics.
- `../../../outputs/autoregressive_transformer_eval*`: local generated manifests;
  reproducible sample arrays are ignored to keep the tree reviewable.
- `figures/`: retained comparison figures; preview arrays and manifests remain
  reproducible local outputs and are ignored.
