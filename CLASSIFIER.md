# Harvey Species Classifier

This branch contains Harvey's trained three-species classifier and the scripts needed to reproduce and evaluate it. Transient `runs/`, local Codex metadata, datasets, notebooks, and local tests are intentionally excluded from the branch scope.

## Structure

```text
configs/spectrogram.json
    Shared 22.05 kHz, 3-second, 128 x 128 log-mel configuration.

scripts/
    01_create_splits.py             Reproduce recording-safe manifests.
    02_build_spectrograms.py        Build the optional full spectrogram cache.
    03_train_classifier.py          Train the residual CNN.
    03_evaluate_classifier.py       Evaluate a checkpoint on the held-out test set.
    03_predict_generated.py         Classify generated WAV or NPY samples.
    03_benchmark_classifier.py      Measure GPU and preprocessing throughput.
    validate_setup.py               Check dependencies, CUDA, data, and one forward pass.

src/bird_song/
    audio.py                        Shared audio/log-mel preprocessing.
    config.py                       Spectrogram configuration loader.
    data.py                         Datasets and Windows-safe DataLoaders.
    runtime.py                      Device and atomic checkpoint helpers.
    classifier/                     CNN model, training, evaluation, prediction, benchmark.

classifier_artifacts/Harvey_classifier/
    best.pt                         Corrected trained checkpoint.
    README.md                       Model card and evaluation results.
```

The split manifests already tracked on the base repository remain the input to the numbered scripts. All segments from the same original recording ID stay in one split to reduce leakage.

## Model

`BirdSongCNN` is a compact residual convolutional neural network with approximately 1.66 million trainable parameters. Its input is a normalized single-channel 128 x 128 log-mel spectrogram. It uses a stride-2 convolutional stem, six residual blocks, global average/maximum pooling, and a two-layer classification head.

## Commands

```powershell
python scripts/validate_setup.py
python scripts/03_train_classifier.py --dry-run
python scripts/03_evaluate_classifier.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --output-dir runs/harvey_classifier/test
python scripts/03_predict_generated.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --input generated_samples --labels-from-parent
```

The classifier is appropriate for target-label evaluation and model-to-model comparisons. It should be combined with blind listening and spectrogram review because a closed-set classifier always chooses one of its three classes, even for unrealistic or out-of-distribution audio.
