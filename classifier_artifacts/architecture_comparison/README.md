# Classifier Architecture Comparison

This folder preserves the complete four-architecture, three-seed validation sweep run on the `Harvey_classifier` branch. It contains all 12 selected checkpoints, per-epoch histories, portable run configurations, aggregate results, and SHA-256 checksums.

## Outcome

| Architecture | Parameters | Runs | Validation accuracy | Validation macro F1 | Mean best epoch |
|---|---:|---:|---:|---:|---:|
| CRNN (CNN-GRU) | 404,451 | 3 | **90.56% +/- 1.39%** | **90.45% +/- 1.45%** | 16.7 |
| Plain CNN | 1,238,691 | 3 | 90.43% +/- 1.74% | 90.33% +/- 1.76% | 14.0 |
| Residual CNN | 1,661,795 | 3 | 87.54% +/- 0.99% | 87.44% +/- 0.97% | 9.3 |
| Depthwise CNN | 137,699 | 3 | 81.63% +/- 2.70% | 81.67% +/- 2.94% | 28.7 |

CRNN and Plain CNN are a validation-quality near tie: CRNN's mean accuracy lead is only 0.13 percentage points and is smaller than seed-to-seed variation. CRNN is the practical candidate because it uses about one-third as many parameters as Plain CNN and was substantially faster in the separate inference benchmark.

The CRNN seed-777 checkpoint achieved the strongest selected-run validation result: 92.10% accuracy and 92.06% macro F1 at epoch 19.

## Important evaluation boundary

These are validation results. The held-out test split was not used to select an architecture. The CRNN seed-777 checkpoint was preselected from this evidence and evaluated exactly once; its packaged report is in [`../selected_crnn`](../selected_crnn/README.md). Do not compare every checkpoint on test data.

The separately published [`Harvey_classifier`](../Harvey_classifier/README.md) residual checkpoint remains the model with the recorded 90.39% held-out test accuracy.

## Contents

- `summary.csv`: architecture-level means and sample standard deviations.
- `runs.csv`: one row per architecture and seed.
- `comparison.md`: generated concise comparison.
- `protocol.json`: common training protocol with repository-relative paths.
- `<architecture>/seed_<seed>/best.pt`: selected checkpoint for that run.
- `<architecture>/seed_<seed>/history.csv`: per-epoch training and validation history.
- `<architecture>/seed_<seed>/config.json`: portable run configuration.
- `SHA256SUMS.txt`: integrity hashes for all checkpoints.

## Reproduce the sweep

From the repository root on a CUDA-capable machine:

```powershell
..\bird_song_venv\Scripts\python.exe scripts\03_compare_classifier_architectures.py `
  --architectures residual_cnn plain_cnn depthwise_cnn crnn `
  --seeds 42 123 777 `
  --output-dir runs\classifier_architectures `
  --epochs 40 --patience 8 --batch-size 64 --workers 4 --device cuda
```

The dataset is intentionally not included in Git. The command expects the recording-safe manifests and local `bird_songs_dataset` described in `protocol.json`.
