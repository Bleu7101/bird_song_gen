# Bird Song Generation

A six-stage pipeline for bird-song generation and evaluation.

## Pipeline

1. Read the dataset and create recording-safe train/validation/test splits.
2. Convert WAV clips to a shared normalized log-mel representation.
3. Train and validate a supervised species classifier.
4. Train a VAE on the shared spectrogram representation.
5. Train a diffusion model on the shared spectrogram representation.
6. Evaluate generated samples with classifier scores, listening tests, and spectrogram inspection.

The target species are American Robin, Northern Cardinal, and Song Sparrow.

## Project structure

```text
configs/
`-- spectrogram.json                         Shared representation for Stages 2-6

scripts/
|-- 01_create_splits.py                      Reproduce recording-safe manifests
|-- 02_build_spectrograms.py                 Build the optional full NPY cache
|-- 03_train_classifier.py                   Train the residual CNN
|-- 03_evaluate_classifier.py                Evaluate the classifier on real held-out audio
|-- 03_benchmark_classifier.py               Benchmark GPU/preprocessing throughput
`-- 06_evaluate_generated.py                 Score generated audio and summarize results

src/bird_song/
|-- audio.py                                 Shared WAV/log-mel preprocessing
|-- config.py                                Spectrogram configuration loader
|-- data.py                                  Dataset and DataLoader code
|-- runtime.py                               Device and checkpoint helpers
`-- classifier/                              Model and Stage 3/6 workflows

classifier_artifacts/Harvey_classifier/
|-- best.pt                                  Trained checkpoint
`-- README.md                                Model card and evaluation results

manifests/                                   Stage 1 outputs
01_dataset_audit.ipynb                       Original data audit and visual exploration
```

## Stage 1: dataset splits

The manifests contain 2,339 training, 519 validation, and 489 test clips. All segments from one original recording ID remain in one split to reduce leakage.

Check deterministic regeneration without changing files:

```powershell
python scripts/01_create_splits.py --dry-run
```

## Stage 2: shared spectrograms

`configs/spectrogram.json` specifies mono 22.05 kHz audio, three-second clips, 128 mel bins, a 1,024-sample FFT, a 512-sample hop, and normalized 128 x 128 outputs in `[-1, 1]`.

Build the complete cache when needed by the VAE/diffusion stages:

```powershell
python scripts/02_build_spectrograms.py
```

The classifier preprocesses WAV files on demand, so a cache is not required to train or evaluate it.

## Stage 3: species classifier

`BirdSongCNN` is a residual CNN with approximately 1.66 million trainable parameters. It uses a convolutional stem, six residual blocks, global average/maximum pooling, and a two-layer classification head.

Train a new run:

```powershell
python scripts/03_train_classifier.py --epochs 40 --batch-size 64 --workers 4
```

Evaluate the trained checkpoint on the real held-out test split:

```powershell
python scripts/03_evaluate_classifier.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --output-dir runs/harvey_classifier/test
```

### Classifier results

| Metric | Result |
|---|---:|
| Best epoch | 8 |
| Validation accuracy | 88.25% |
| Test accuracy | 90.39% |
| Test macro F1 | 90.44% |

Per-species test F1 is 91.23% for American Robin, 90.16% for Northern Cardinal, and 89.95% for Song Sparrow. Full results, limitations, checksum, and the confusion matrix are in `classifier_artifacts/Harvey_classifier/README.md`.

## Stages 4 and 5: generation

The VAE and diffusion implementations should import `bird_song.audio` and use `configs/spectrogram.json`. They should not copy preprocessing code from the audit notebook or introduce another normalization convention.

## Stage 6: generated-audio evaluation

Organize labeled generated samples by their intended species:

```text
generated_samples/
|-- american_robin/
|-- northern_cardinal/
`-- song_sparrow/
```

Run:

```powershell
python scripts/06_evaluate_generated.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --input generated_samples --labels-from-parent
```

This writes:

- Per-file predictions, confidence, and class probabilities in CSV format.
- A JSON summary containing sample count, mean confidence, predicted-class counts, overall target-label accuracy, and per-target accuracy.

Classifier scores are not a complete realism metric. Generated audio is out-of-distribution, and this closed-set model must choose one of its three classes even for noise. The final report should combine Step 6 classifier results with blind listening and spectrogram inspection.
