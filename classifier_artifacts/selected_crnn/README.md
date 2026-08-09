# Selected CRNN classifier

This is the current classifier package for new real-audio and generated-sample
evaluation commands.

## Checkpoint

- File: `best.pt`
- Architecture: CRNN (CNN followed by a bidirectional GRU)
- Seed: 777
- Selected epoch: 19
- Trainable parameters: 404,451
- SHA-256: `60525C2AB3EE2B9FC24D12DB6F591010BFA36FAC2D2804D6946C7A6021121806`
- Validation accuracy: 92.10%
- Validation macro F1: 92.06%

The checkpoint was selected before looking at the held-out test split. The
unchanged source run remains at
`../architecture_comparison/crnn/seed_777/best.pt`.

## One held-out evaluation

The checkpoint was evaluated once on the recording-ID-isolated historical v1
489-clip test split. A later exact-content audit found no train/test or
validation/test duplicate, so none of the identified byte-identical leakage
reaches this held-out result. The complete machine-readable report is in
`metrics.json`; predictions use dataset-relative paths in `predictions.csv`.

| Metric | Result |
|---|---:|
| Accuracy | **89.98%** |
| Macro precision | 90.25% |
| Macro recall | 90.18% |
| Macro F1 | **90.16%** |

| Species | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| American Robin | 92.00% | 95.17% | 93.56% | 145 |
| Northern Cardinal | 92.05% | 85.80% | 88.82% | 162 |
| Song Sparrow | 86.70% | 89.56% | 88.11% | 182 |

Confusion matrix, rows true and columns predicted:

|  | American Robin | Northern Cardinal | Song Sparrow |
|---|---:|---:|---:|
| American Robin | 138 | 2 | 5 |
| Northern Cardinal | 3 | 139 | 20 |
| Song Sparrow | 9 | 10 | 163 |

See [`confusion_matrix.png`](confusion_matrix.png) and
[`confusion_matrix.csv`](confusion_matrix.csv) for the rendered and raw
versions. `SHA256SUMS.txt` records the package integrity hashes.

## Reproduce evaluation

From the repository root, with the local dataset available:

```powershell
$env:PYTHONPATH = "src"
$py = "..\bird_song_venv\Scripts\python.exe"
& $py scripts/02_build_spectrograms.py --output-dir artifacts/spectrograms
& $py scripts/03_evaluate_classifier.py `
  --checkpoint classifier_artifacts/selected_crnn/best.pt `
  --output-dir runs/selected_crnn_test `
  --batch-size 128 --workers 0 --device auto
```

The legacy residual package remains available for reproducing the earlier
transformer and generator reports; its held-out result is not interchangeable
with this CRNN evaluation.

## Role in the first augmentation evaluation

The VAE-v3/diffusion augmentation report uses this 89.98%-accuracy, 90.16%-macro-F1
checkpoint as a descriptive historical reference. It is not a matched control:
this model is one validation-selected seed trained through the earlier
on-the-fly WAV path with stochastic transforms, while each augmented condition
is a three-seed mean trained from fixed cached real spectrograms. Therefore the
reported negative mean deltas are no demonstrated augmentation gain, not proof
that synthetic data is harmful. See
[`reports/crnn_synthetic_augmentation_2026-08-09`](../../reports/crnn_synthetic_augmentation_2026-08-09/README.md).
