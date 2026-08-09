# `BigVGAN_decode`: future audio-decoder research

This branch is reserved for future work testing different decoders and
vocoders that convert generated spectrograms into audio waveforms. The current
contents are the first recorded experiment: they use the exact 80-band,
22.05 kHz mel contract required by
`nvidia/bigvgan_v2_22khz_80band_256x`, retrain the conditional WGAN-GP and
Transformer for that representation, and decode generated mels with the frozen
BigVGAN generator.

This is a decoder-research branch, not an additional repository-wide generator
family. `main` remains the primary classifier, VAE v1/v2/v3, Transformer, and
WGAN-GP review branch. Diffusion models remain on the separate `Diffusion`,
`Difussion`, and `diffusion_vincent` branches.

The `256x` model suffix is its waveform upsampling ratio, not a restriction to
256 frames. This experiment uses 256 frames because the official configuration
uses a 65,536-sample segment and hop length 256.

## Setup

The dataset is not committed. It must contain `wavfiles/` and
`bird_songs_metadata.csv` under `bird_songs_dataset/`. The current Windows
worktree uses the existing sibling `bird_song_venv` environment.

```powershell
$py = "..\bird_song_venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pip install -e .
& $py -B -m pytest -p no:cacheprovider -p no:anyio -q
```

Fetch the frozen generator (the command is idempotent):

```powershell
& $py scripts/01_fetch_bigvgan.py
```

## Recorded BigVGAN decoder baseline

Build the raw BigVGAN mel cache. The scaler is fitted on training clips only.

```powershell
& $py scripts/02_build_bigvgan_mels.py --device auto --overwrite
```

Run the balanced held-out comparison (30 clips per species):

```powershell
& $py scripts/03_test_bigvgan.py `
  --per-species 30 --device auto --overwrite
```

The ignored output contains original/preprocessed, Griffin-Lim, and BigVGAN
WAVs, plus `decoder_summary.json`, `decoder_metrics.csv`, and a listening
manifest. The automatic decoder pass requires finite 65,536-sample outputs,
at most 0.1% clipping, and lower mean multi-resolution STFT spectral and
log-magnitude errors than Griffin-Lim. Listening remains necessary.

## WGAN-GP test

Run the one-batch CUDA wiring smoke test first:

```powershell
& $py scripts/04_train_wgan_gp.py `
  --epochs 1 --max-batches 1 --eval-batches 1 `
  --output-dir runs/wgan_gp_smoke --device auto --overwrite
```

Then run the recorded pilot defaults:

```powershell
& $py scripts/04_train_wgan_gp.py `
  --epochs 20 --batch-size 32 --critic-steps 2 --workers 0 `
  --output-dir runs/wgan_gp_bigvgan --device auto --overwrite
```

Generate eight samples per species and decode them with BigVGAN:

```powershell
& $py scripts/05_generate_and_decode.py `
  --checkpoint runs/wgan_gp_bigvgan/best_generator.pt `
  --samples-per-species 8 --device auto --overwrite
```

`best_generator.pt`, the resumable `last.pt`, caches, and generated audio are
ignored. The generator-only checkpoint stores its model shape, classes, mel
contract, training scaler, seed, and validation-selection metrics.

## Generator retraining matrix

The recorded comparison uses the same cache and validation split for seeds
42, 123, and 777:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_bigvgan_matrix.ps1
```

The WGAN baseline uses equal `1e-4` learning rates and two critic steps. The
stability candidate uses generator `5e-5`, critic `1e-4`, and three critic
steps. The Transformer comparison uses rectangular 16×16 and 8×16 patches.
Checkpoints are selected from validation evidence only; the compact recorded
matrix is [reports/bigvgan_matrix_report.md](reports/bigvgan_matrix_report.md).

The selected models are the stability WGAN (seed 42) and the 8×16 Transformer
(seed 42). The small publication bundle, model cards, hashes, configs, and a
balanced listening set are in [reports/canonical_models](reports/canonical_models).

Generate and evaluate a Transformer set at a requested temperature:

```powershell
& $py -B scripts/07_generate_transformer.py `
  --checkpoint runs/experiments/transformer_8x16_seed42/best.pt `
  --temperature 0.8 --samples-per-species 8 --device cuda `
  --output-dir runs/final_evaluations/transformer_selected/temp_0.8 --overwrite
& $py -B scripts/08_evaluate_generators.py `
  --generated-manifest runs/final_evaluations/transformer_selected/temp_0.8/generated_manifest.csv `
  --output-json reports/transformer_evaluation.json
```

## Evidence boundary

This branch contains only the BigVGAN decoder experiment and the WGAN-GP and
rectangular Transformer variants retrained for its `80 x 256` contract. VAE
v1/v2/v3 artifacts are on `main`; diffusion models are on the separate
diffusion branches. The generated-audio report therefore
checks shape, finite values, mel saturation, waveform length, clipping, RMS,
silence, detail ratios, and diversity. Those checks establish that the WGAN →
BigVGAN handoff works; they do not prove that the generated sounds are realistic
or correctly conditioned. Use the local listening manifest for that judgment.
The frozen legacy residual CNN and CRNN are conditioning diagnostics only, not
perceptual-realism scores.

## Future decoder work

Future experiments on this branch should compare alternative decoder/vocoder
paths using the same source spectrograms, train/validation/test boundaries,
waveform-validity gates, and listening manifests. Objective spectral metrics
and classifier agreement remain diagnostics; decoder selection also requires
same-source listening review. The recorded BigVGAN result is a baseline for
those comparisons, not a final decoder selection.
