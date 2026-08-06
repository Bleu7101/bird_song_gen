# Classifier architecture comparison

All architectures used the same recording-safe train/validation manifests, preprocessing, optimizer settings, early-stopping rule, and seed set. The held-out test split was not used for architecture selection.

Seeds: `42, 123, 777`. Maximum epochs: 40. Early-stopping patience: 8. Batch size: 64. Learning rate: 0.0003. Weight decay: 0.0001. Width: 32. Dropout: 0.3.

Each run's selected epoch is the one with the highest validation accuracy. Values are mean +/- sample standard deviation when at least two seeds were run.

| Architecture | Parameters | Runs | Validation accuracy | Validation macro F1 | Mean best epoch |
|---|---:|---:|---:|---:|---:|
| crnn | 404,451 | 3 | 90.56% +/- 1.39% | 90.45% +/- 1.45% | 16.7 |
| plain_cnn | 1,238,691 | 3 | 90.43% +/- 1.74% | 90.33% +/- 1.76% | 14.0 |
| residual_cnn | 1,661,795 | 3 | 87.54% +/- 0.99% | 87.44% +/- 0.97% | 9.3 |
| depthwise_cnn | 137,699 | 3 | 81.63% +/- 2.70% | 81.67% +/- 2.94% | 28.7 |
