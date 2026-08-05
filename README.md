# `decoder_test`: BigVGAN-compatible bird-song experiment

This branch is intentionally small. It changes the real-audio representation to
the exact 80-band, 22.05 kHz mel contract used by
`nvidia/bigvgan_v2_22khz_80band_256x`, retrains the conditional WGAN-GP on that
representation, and decodes generated mels with the frozen BigVGAN generator.

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

## Real-audio decoder test

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

## Evidence boundary

This branch intentionally removes the classifier, transformer, VAE, diffusion,
notebook, and historical output trees. The generated-audio report therefore
checks shape, finite values, mel saturation, waveform length, clipping, RMS,
silence, detail ratios, and diversity. Those checks establish that the WGAN →
BigVGAN handoff works; they do not prove that the generated sounds are realistic
or correctly conditioned. Use the local listening manifest for that judgment.
