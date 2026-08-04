# Bird Song Generation

A seven-stage pipeline for bird-song generation and evaluation.

## Pipeline

1. Read the dataset and create recording-safe train/validation/test splits.
2. Convert WAV clips to a shared normalized log-mel representation.
3. Train and validate a supervised species classifier.
4. Train a VAE on the shared spectrogram representation.
5. Train a diffusion model on the shared spectrogram representation.
6. Train a conditional autoregressive transformer that generates log-mel images patch by patch.
7. Evaluate generated samples with classifier scores, listening tests, and spectrogram inspection.

The target species are American Robin, Northern Cardinal, and Song Sparrow.

## Project structure

```text
configs/
|-- spectrogram.json                         Protected 128 x 128 classifier representation
|-- transformer.json                         Autoregressive generator architecture
|-- vocoder_spectrogram.json                 Exact full-band BigVGAN mel contract
|-- vocoder_spectrogram_fmax8k.json          One allowed fallback vocoder contract
`-- vocoder_vae.json                         Rectangular spatial VAE v2 architecture

notebooks/
|-- 01_dataset_audit.ipynb                   Stage 1 data audit and split exploration
|-- 02_preprocess_logmel.ipynb               Stage 2 log-mel exploration and visual checks
|-- 03_classifier.ipynb                      Stage 3 architecture experiment and visual analysis
|-- 04_conditional_vae.ipynb                 Stage 4 conditional VAE experiment
`-- 06_autoregressive_transformer.ipynb      Stage 6 transformer generation experiment

scripts/
|-- 01_create_splits.py                      Reproduce recording-safe manifests
|-- 02_build_spectrograms.py                 Build the optional full NPY cache
|-- 03_train_classifier.py                   Train a selected classifier architecture
|-- 03_compare_classifier_architectures.py   Run a controlled architecture comparison
|-- 03_evaluate_classifier.py                Evaluate the classifier on real held-out audio
|-- 03_benchmark_classifier.py               Benchmark GPU/preprocessing throughput
|-- 06_train_transformer.py                  Train the autoregressive transformer generator
|-- 06_generate_transformer.py               Generate conditional log-mel images
|-- 06_evaluate_generated.py                 Legacy generated-sample evaluation entry point
|-- 07_evaluate_generated.py                 Score generated samples and summarize results
|-- 08_fetch_bigvgan.py                      Fetch only the frozen generator and source
|-- 08_evaluate_vocoder_gate.py              Run the real-audio reconstruction gate
|-- 08_build_vocoder_spectrograms.py         Build isolated raw 80 x 256 log-mels
|-- 09_train_vocoder_vae.py                  Train the rectangular spatial VAE v2
|-- 09_generate_vocoder_vae.py               Fit priors and generate 64 WAVs per species
|-- 10_evaluate_vocoder_vae.py               Run the final VAE audio gate
|-- 10_prepare_listening_study.py            Build a blinded, balanced listening pack
`-- 10_score_listening_study.py              Validate and summarize listener ratings

src/bird_song/
|-- audio.py                                 Shared WAV/log-mel preprocessing
|-- config.py                                Spectrogram configuration loader
|-- data.py                                  Dataset and DataLoader code
|-- runtime.py                               Device and checkpoint helpers
|-- classifier/                              Model and classifier-evaluation workflows
|-- vae/                                     Stage 4 package boundary
|-- diffusion/                               Stage 5 package boundary
`-- transformer/                             Stage 6 model, cache loader, training, and generation

classifier_artifacts/
|-- Harvey_classifier/                       Published residual model and held-out results
`-- architecture_comparison/                 Complete 4-architecture x 3-seed validation sweep

manifests/                                   Stage 1 outputs
```

## Stage 1: dataset splits

The manifests contain 2,339 training, 519 validation, and 489 test clips. All segments from one original recording ID remain in one split to reduce leakage.

Check deterministic regeneration without changing files:

```powershell
python scripts/01_create_splits.py --dry-run
```

## Stage 2: shared spectrograms

`configs/spectrogram.json` specifies mono 22.05 kHz audio, three-second clips, 128 mel bins, a 1,024-sample FFT, a 512-sample hop, and normalized 128 x 128 outputs in `[-1, 1]`.

Build the complete cache when needed by the VAE, diffusion, and transformer stages:

```powershell
python scripts/02_build_spectrograms.py
```

The classifier preprocesses WAV files on demand, so a cache is not required to train or evaluate it. Use `notebooks/02_preprocess_logmel.ipynb` for exploration and visual checks; use the Stage 2 script for the shared cache consumed by later stages.

## Stage 3: species classifier

The original `BirdSongCNN` is a residual CNN with approximately 1.66 million trainable parameters. It uses a convolutional stem, six residual blocks, global average/maximum pooling, and a two-layer classification head.

Train a new run:

```powershell
python scripts/03_train_classifier.py --epochs 40 --batch-size 64 --workers 4
```

Choose an individual architecture with `--architecture`. The implemented alternatives test different inductive biases and model sizes:

| Architecture | Main idea | Parameters at width 32 |
|---|---|---:|
| `residual_cnn` | Six residual convolution blocks | 1,661,795 |
| `plain_cnn` | VGG-style convolution stack without skip connections | 1,238,691 |
| `crnn` | Convolutions followed by a bidirectional GRU over time | 404,451 |
| `depthwise_cnn` | MobileNet-style depthwise-separable convolutions | 137,699 |

Run the controlled comparison with three seeds per architecture:

```powershell
python scripts/03_compare_classifier_architectures.py --epochs 40 --patience 8 --seeds 42 123 777
```

The comparison holds the data splits, preprocessing, seeded data order, width, dropout, optimizer, learning rate, and early-stopping rule fixed. It writes every run separately plus `protocol.json`, `runs.csv`, `summary.csv`, and a presentation-ready `comparison.md` containing mean and sample standard deviation for validation accuracy and macro F1. Parameter counts are reported because the architectures deliberately span high- and low-capacity models. Use validation results to select the architecture, then evaluate only the selected checkpoint on the test split; repeatedly choosing models on test accuracy would leak test information.

The completed 12-run sweep ranked CRNN first at **90.56% +/- 1.39% validation accuracy** and **90.45% +/- 1.45% validation macro F1**, narrowly ahead of Plain CNN at 90.43% +/- 1.74% accuracy. That quality gap is smaller than seed variation, so CRNN is the practical candidate based on its substantially smaller model and faster measured inference. All checkpoints, histories, portable configs, aggregate results, and integrity hashes are preserved in [`classifier_artifacts/architecture_comparison`](classifier_artifacts/architecture_comparison/README.md). These alternative checkpoints have not been evaluated on the held-out test split.

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

## Stages 4-6: generation

The VAE, diffusion, and transformer use the same normalized representation from `configs/spectrogram.json`. The transformer is a species-conditional autoregressive image generator, not a classifier: it converts each 128 x 128 spectrogram into 64 time-major 16 x 16 patches and predicts a Gaussian distribution for each next patch using causal self-attention.

Train the transformer on the Stage 2 cache:

```powershell
python scripts/06_train_transformer.py --epochs 60 --batch-size 32 --workers 4 --device cuda
```

Generate eight log-mel images per species from the selected checkpoint:

```powershell
python scripts/06_generate_transformer.py --checkpoint runs/transformer_generator/best.pt --samples-per-species 8 --temperature 0.8 --device cuda
```

The generator writes normalized `.npy` images under species-named directories, a `generated_manifest.csv`, and a visual `conditional_samples.png`. Use `notebooks/06_autoregressive_transformer.ipynb` for patch-order visualization, gated training and generation, loss curves, real-versus-generated comparisons, and diversity diagnostics.

The completed 4.95M-parameter transformer run, checkpoint, temperature sweep, classifier interoperability evaluation, previews, and technical verdict are documented in [`runs/transformer_generator/README.md`](runs/transformer_generator/README.md). The model trained cleanly and produces classifier-readable spectrograms, but the generated images remain blurrier and less structured than real bird calls; treat it as a working baseline rather than a realism-ready generator.

## Stage 7: generated-sample evaluation

Organize labeled generated samples by their intended species:

```text
generated_samples/
|-- american_robin/
|-- northern_cardinal/
`-- song_sparrow/
```

Run:

```powershell
python scripts/07_evaluate_generated.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --input generated_samples --labels-from-parent
```

This writes:

- Per-file predictions, confidence, and class probabilities in CSV format.
- A JSON summary containing sample count, mean confidence, predicted-class counts, overall target-label accuracy, and per-target accuracy.

Classifier scores are not a complete realism metric. Generated audio is out-of-distribution, and this closed-set model must choose one of its three classes even for noise. The final report should combine Stage 7 classifier results with blind listening and spectrogram inspection.

## Vocoder-first audio branch

This branch is deliberately separate from the protected `128 x 128` classifier/generation branch. It never resizes old generated arrays or changes `configs/spectrogram.json`, the classifier checkpoint, published metrics, or existing generated samples. Its representation exactly matches `nvidia/bigvgan_v2_22khz_80band_256x`: mono 22,050 Hz, 65,536 samples, a 1,024-sample FFT/window, a 256-sample hop, 80 Slaney mel bands from 0 Hz to Nyquist, and 256 frames. The frozen vocoder uses `use_cuda_kernel=False`; preprocessing and inference prefer CUDA automatically when it is available.

Install dependencies and fetch only the official frozen generator plus inference source:

```powershell
python -m pip install -r requirements.txt
python scripts/08_fetch_bigvgan.py
```

### Gate 1: prove the representation can be heard

Run 30 held-out clips per species through the original-audio control, a Griffin-Lim baseline, and the full-band BigVGAN reconstruction:

```powershell
python scripts/08_evaluate_vocoder_gate.py --device cuda
```

The automatic gate requires finite length-correct waveforms, no material clipping, and BigVGAN classifier accuracy within five percentage points of the original-WAV control. `runs/vocoder_gate/gate_summary.json` remains `awaiting_listening` until the group supplies pilot ratings with median BigVGAN bird-likeness at least 3/5. To prepare and score a blinded pilot:

```powershell
python scripts/10_prepare_listening_study.py `
  --condition original=runs/vocoder_gate/audio/original `
  --condition griffin_lim=runs/vocoder_gate/audio/griffin_lim `
  --condition bigvgan=runs/vocoder_gate/audio/bigvgan `
  --output-dir runs/vocoder_gate/listening

python scripts/10_score_listening_study.py `
  --key runs/vocoder_gate/listening/blind_key.csv `
  --responses runs/vocoder_gate/listening/responses/*.csv `
  --output-dir runs/vocoder_gate/listening/results

python scripts/08_evaluate_vocoder_gate.py `
  --device cuda --overwrite `
  --listening-ratings runs/vocoder_gate/listening/results/combined_listening_ratings.csv
```

If the full-band automatic or listening gate fails, run the allowed 8 kHz fallback once on the same clips:

```powershell
python scripts/08_fetch_bigvgan.py `
  --model-id nvidia/bigvgan_v2_22khz_80band_fmax8k_256x

python scripts/08_evaluate_vocoder_gate.py `
  --device cuda `
  --vocoder-config configs/vocoder_spectrogram_fmax8k.json `
  --output-dir runs/vocoder_gate_fmax8k
```

If neither checkpoint passes, stop here and report the reconstruction ceiling alongside the existing classifier-readable fallback. Do not train or fine-tune a vocoder in this cycle.

### Train only after Gate 1 passes

The cache stores raw, unclipped BigVGAN log-mels. Its global mean and standard deviation are fitted from training examples only and are embedded in the VAE checkpoint so inference can reverse normalization exactly:

```powershell
python scripts/08_build_vocoder_spectrograms.py --device cuda
python scripts/09_train_vocoder_vae.py --device cuda --epochs 60 --batch-size 32 --workers 4
python scripts/09_generate_vocoder_vae.py `
  --device cuda --samples-per-species 64 --temperature 0.7
```

Training uses seed 42, a 15-epoch KL warmup, early stopping, and the ported detail-aware spatial VAE v2. Four factor-two stages map `[1, 80, 256]` to a `[16, 5, 16]` latent map. Generation fits a per-species aggregated posterior prior on the training split before sampling.

### Gate 2: determine whether the VAE is audio-ready

Run the classifier through its normal WAV preprocessing path for the original, vocoder ceiling, deterministic VAE reconstruction, and generated-waveform conditions:

```powershell
python scripts/10_evaluate_vocoder_vae.py --device cuda
```

The automatic VAE gate requires deterministic reconstruction accuracy within ten points of the BigVGAN ceiling and generated-waveform target accuracy above 50%. Prepare the requested 24-clip balanced study with two clips per species per condition, collect approximately 8-12 complete responses, and score them:

```powershell
python scripts/10_prepare_listening_study.py `
  --condition original=runs/vocoder_vae/evaluation/audio/original `
  --condition bigvgan=runs/vocoder_vae/evaluation/audio/bigvgan `
  --condition vae_reconstruction=runs/vocoder_vae/evaluation/audio/vae_reconstruction `
  --condition vae_generated=outputs/vocoder_vae/generated_audio

python scripts/10_score_listening_study.py `
  --responses runs/listening_study/responses/*.csv

python scripts/10_evaluate_vocoder_vae.py `
  --device cuda --overwrite `
  --listening-ratings runs/listening_study/results/combined_listening_ratings.csv
```

Only a median generated bird-likeness score of at least 3/5 changes the final status to `audio_ready`. The reports also include paired multi-resolution STFT error and Gaussian Frechet distance in the published residual-classifier embedding space. That latter value is explicitly a domain-specific, FAD-style diagnostic, not standard VGGish FAD, and is never used alone. Direct classification of the old 128 x 128 generated images remains available as a secondary diagnostic in Stage 7; it is not applied to the incompatible 80 x 256 vocoder representation.

### Twelve-day ownership handoff

Assign one person to each owner role before starting the next gate-dependent step:

| Days | Owner role | Deliverable |
|---|---|---|
| 1-2 | Vocoder/cache owner | Frozen adapter, 90-clip reconstruction gate, and pilot listening pack |
| 3-4 | Vocoder/cache owner | Raw 80 x 256 cache plus training-only normalization metadata and contract checks |
| 5-8 | VAE owner | Gated 60-epoch-maximum spatial VAE run and selected checkpoint |
| 9-10 | Evaluation/report owner | 64 decoded samples per species, WAV-path classifier results, paired STFT errors, and real-vs-real-calibrated embedding distances |
| 11-12 | Evaluation/report owner | Blinded 24-clip study, scored ratings, final gate status, and fallback conclusion |
