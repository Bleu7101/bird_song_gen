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
|-- spectrogram.json                         Shared representation for Stages 2-7
|-- transformer.json                         Continuous autoregressive baseline
|-- wgan_gp.json                             Conditional WGAN-GP
|-- vqgan.json                               Adversarial tokenizer/decoder
|-- token_transformer.json                   Transformer over VQGAN tokens
`-- latent_diffusion.json                    Diffusion over VQGAN latents

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
|-- 08_train_wgan_gp.py                      Train the sharp direct-GAN baseline
|-- 09_train_vqgan.py                        Train the adversarial tokenizer/decoder
|-- 09_train_token_transformer.py            Train a transformer over discrete tokens
|-- 10_train_latent_diffusion.py             Train diffusion over VQGAN latents
|-- 11_evaluate_generators.py                Compare detail, validity, and diversity
`-- 12_decode_generated_audio.py             Fixed Griffin-Lim listening decoder

src/bird_song/
|-- audio.py                                 Shared WAV/log-mel preprocessing
|-- config.py                                Spectrogram configuration loader
|-- data.py                                  Dataset and DataLoader code
|-- runtime.py                               Device and checkpoint helpers
|-- classifier/                              Model and classifier-evaluation workflows
|-- vae/                                     Stage 4 package boundary
|-- diffusion/                               Stage 5 package boundary
|-- transformer/                             Stage 6 continuous-patch baseline
`-- generation/                              WGAN, VQGAN/token, diffusion, audio, and metrics

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

The generator recovery branch adds a conditional WGAN-GP, a VQGAN/token
transformer path, and latent diffusion without changing the classifier. The
first full-data WGAN pilot produced sharper, diverse spectrograms and 84.90%
target agreement with the residual classifier, but only 40.63% with the CRNN.
The evidence, histories, curated audio, and next-step verdict are in
[`reports/generator_pilot_2026-08-04`](reports/generator_pilot_2026-08-04/README.md).

The recommended next experiment is WGAN-GP v2 with limited-data discriminator
augmentation and checkpoint selection based on realistic detail, diversity,
and cross-classifier consistency. The current rule favors maximum detail and
can select noisy over-sharp checkpoints.

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
