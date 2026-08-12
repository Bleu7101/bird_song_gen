# Legacy Residual CNN Baseline Model Card

The directory name `Harvey_classifier` is a historical artifact label, not an
active Git branch. The repository's primary branch is now `main`.

This is the original held-out-tested residual CNN checkpoint. It remains in
the repository as historical real-audio classifier evidence and because VAE-v3
training used it as a frozen teacher for feature and label-consistency losses.
It is not used for maintained generated-sample evaluation. The
validation-selected CRNN is packaged separately under `../selected_crnn/`.

## Intended use

Reproduce the historical three-species real-audio classifier result or the
recorded VAE-v3 training-teacher dependency. Use the selected CRNN for current
generated-sample evaluation.

## Checkpoint

- File: `best.pt` (legacy baseline; not the current selected classifier)
- Architecture: residual CNN, approximately 1.66 million trainable parameters
- Selected epoch: 8
- Validation accuracy: 88.25%
- Preprocessing: stored inside the checkpoint and mirrored by `configs/spectrogram.json`

## Held-out test evaluation

The historical v1 test manifest contains 489 clips isolated from training and
validation by original recording ID. A later exact-content audit also found no
duplicate reaching test; the nine identified cross-split duplicate clips affect
training and validation only.

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
- The classifier cannot reject noise or unknown bird species.
