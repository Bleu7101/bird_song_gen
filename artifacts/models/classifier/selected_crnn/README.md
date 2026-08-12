# Selected CRNN classifier

This is the current classifier package for new real-audio and generated-sample
evaluation commands.

## Checkpoint

- File: `best.pt`
- Architecture: CRNN (CNN followed by a bidirectional GRU)
- Seed: 777
- Selected epoch: 19
- Trainable parameters: 404,451
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
versions.

## Reproduce evaluation

From the repository root, with the local dataset available:

```powershell
$env:PYTHONPATH = "src"
$py = "..\bird_song_venv\Scripts\python.exe"
& $py scripts/02_build_spectrograms.py --output-dir artifacts/spectrograms
& $py scripts/03_evaluate_classifier.py `
  --checkpoint artifacts/models/classifier/selected_crnn/best.pt `
  --output-dir runs/selected_crnn_test `
  --batch-size 128 --workers 0 --device auto
```

The legacy residual package remains available as historical real-audio
classifier evidence; it is not used for maintained generated-sample scoring.

## Role in generator evaluation

This checkpoint is the sole classifier used to score generated VAE-v3,
diffusion, Transformer, and WGAN-GP samples. Target-label accuracy and macro F1
measure compatibility with this closed-set classifier, not perceptual realism.
The VAE-v3/diffusion checkpoint report also uses this model's frozen feature
space for Fréchet distance, feature precision and recall, density, coverage,
diversity, and nearest-neighbor copying-risk diagnostics.
