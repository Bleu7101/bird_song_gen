# Evaluation and future work - 1.5-minute script

Let's go into our model evaluations.

The first graph tests three generation seeds. VAE-v3 was stable: 96.9% macro F1 with only 0.5-point variation. Diffusion was unstable: two seeds reached about 93%, but seed 42 fell to 19%.

The table confirms this in CRNN feature space: VAE is closer to real data and covers more variation.

Can generated data help when labels are scarce? With 50 real spectrograms per species, the CRNN scored 84.75% macro F1. Adding 200 VAE samples raised it to 87.69%: a 2.93-point gain in eight of nine runs. Diffusion added only 0.42 points and was inconsistent.

Overall, generated data can help, but stability matters. Future work is truly rare species and better vocoders for audio conversion. This concludes the evaluation. Thank you; I'm happy to take questions.

This is 130 words: roughly 78 seconds at 100 words per minute, leaving a pause margin.

## Suggested table under the left plot

Use this compact table as **Generator quality (frozen CRNN feature space)**. The arrows indicate the preferred direction; it is not an audio-realism score.

| Metric | VAE-v3 | Diffusion |
|---|---:|---:|
| Classifier-feature distance (lower is better) | 22.58 | 210.06 |
| k=5 manifold precision (higher is better) | 85.64% | 39.53% |
| k=5 manifold coverage (higher is better) | 75.61% | 32.62% |

Small caption: `Three generation seeds; 200 samples per species; metrics computed against real content-safe test features.`

## Image sources gathered

- `part1_conditioning_by_seed.png`: generator conditioning across seeds.
- `part3_paired_test_deltas.png`: matched low-resource augmentation effects.
- `evaluation_future_work_one_slide.png`: ready-to-use composite slide image.
