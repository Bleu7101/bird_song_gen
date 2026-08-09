# Legacy Residual CNN Baseline Model Card

The directory name `Harvey_classifier` is a historical artifact label, not an
active Git branch. The repository's primary branch is now `main`.

This is the original held-out-tested residual CNN checkpoint. It remains in
the repository so earlier transformer and generator evaluations stay exactly
reproducible. The validation-selected CRNN is now packaged separately under
`../selected_crnn/`.

## Intended use

Classify real or generated three-second samples as American Robin, Northern Cardinal, or Song Sparrow. In Stage 7, use the output as one generated-audio evaluation signal, not as a standalone realism score.

## Checkpoint

- File: `best.pt` (legacy baseline; not the current selected classifier)
- SHA-256: `7BE034908A80EF23EC83BF6B1B731B803EE13D6DF1CC548429B35C2DD3E35718`
- Architecture: residual CNN, approximately 1.66 million trainable parameters
- Selected epoch: 8
- Validation accuracy: 88.25%
- Preprocessing: stored inside the checkpoint and mirrored by `configs/spectrogram.json`

## Held-out test evaluation

The test manifest contains 489 clips isolated from training and validation by original recording ID.

| Metric | Result |
|---|---:|
| Accuracy | 90.39% |
| Macro precision | 90.80% |
| Macro recall | 90.24% |
| Macro F1 | 90.44% |

| Species | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| American Robin | 92.86% | 89.66% | 91.23% | 145 |
| Northern Cardinal | 92.81% | 87.65% | 90.16% | 162 |
| Song Sparrow | 86.73% | 93.41% | 89.95% | 182 |

Confusion matrix, with rows as true labels and columns as predictions:

|  | American Robin | Northern Cardinal | Song Sparrow |
|---|---:|---:|---:|
| American Robin | 130 | 1 | 14 |
| Northern Cardinal | 8 | 142 | 12 |
| Song Sparrow | 2 | 10 | 170 |

## Limitations

- Roughly one in ten real held-out clips is misclassified.
- Generated audio is out-of-distribution and may produce overconfident predictions.
- The classifier cannot reject noise or unknown bird species.
- Target-label rate should be reported alongside listening tests and spectrogram inspection.

## Stage 7 command

```powershell
python scripts/07_evaluate_generated.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --input generated_samples --labels-from-parent
```

The command writes per-file probabilities to CSV and an aggregate JSON summary containing sample count, mean confidence, predicted-class counts, and target-label accuracy when generated samples are stored in species-named parent directories.
