# `decoder_test` recorded evidence

Recorded locally on 2026-08-06 from the `decoder_test` worktree at
`C:\Users\Harvey\Documents\Coding\bird_song_gen_decoder_test`.

## Reproduction commands

```powershell
Set-Location C:\Users\Harvey\Documents\Coding\bird_song_gen_decoder_test
$py = "..\bird_song_venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pip install -e .
& $py -B scripts/01_fetch_bigvgan.py
& $py -B scripts/02_build_bigvgan_mels.py --device cuda --overwrite
& $py -B scripts/03_test_bigvgan.py --per-species 30 --griffin-lim-iterations 32 --device cuda --output-dir runs/bigvgan_real_test --overwrite
& $py -B scripts/04_train_wgan_gp.py --epochs 1 --max-batches 1 --eval-batches 1 --batch-size 2 --device cuda --output-dir runs/wgan_gp_smoke --overwrite
& $py -B scripts/04_train_wgan_gp.py --epochs 20 --batch-size 32 --critic-steps 2 --workers 0 --device cuda --output-dir runs/wgan_gp_bigvgan --overwrite
& $py -B scripts/05_generate_and_decode.py --checkpoint runs/wgan_gp_bigvgan/best_generator.pt --samples-per-species 8 --device cuda --output-dir runs/wgan_gp_bigvgan_audio --overwrite
& $py -B -m pytest -p no:cacheprovider -p no:anyio -q
```

`01_fetch_bigvgan.py` was idempotent because the local snapshot was already
available through the worktree's `external` junction.

## Contracts and hashes

| Item | Recorded value |
|---|---|
| BigVGAN model | `nvidia/bigvgan_v2_22khz_80band_256x` |
| Frontend | 22,050 Hz, 65,536 samples, FFT/window 1,024, hop 256, reflect padding, 80-band Slaney mel, `fmin=0`, full Nyquist, natural log |
| Array contract | `[80, 256]` raw log-mel; WGAN tensor `[batch, 1, 80, 256]`; decoder waveform 65,536 samples |
| BigVGAN generator SHA-256 | `E95BA25972D3DE0628D99CD156E9315A9C018899BF739988959EBE3544080CED` |
| BigVGAN config SHA-256 | `88A1F47ACF747DB0B21E97A389D838566147F7A5464583FF5C8D819D870F03EE` |
| Selected generator SHA-256 | `41AAC48BD0F80C03CABA085E1746D889325ED4FBCA46FACD5494AE40F0D332D6` |

The selected WGAN checkpoint is generator-only and records the model shape,
class order, vocoder contract, scaler, seed, epoch, and selection metrics.
The resumable `last.pt`, both model files, the BigVGAN snapshot, cache, and
WAVs are deliberately ignored and remain local.

## Recorded results

### Frontend, cache, and tests

- The focused suite passed: **6 passed**.
- The one-batch CUDA WGAN smoke test completed and wrote a checkpoint.
- The cache contains 3,347 clips and raw arrays of shape `(80, 256)`.
- The training-only global scaler is `[-11.512925, 2.108787]` over 47,902,720 values.
- Validation: 519 clips, range `[-11.512925, 1.659673]`, 0 values outside the training bounds.
- Test: 489 clips, range `[-11.512925, 1.968577]`, 0 values outside the training bounds.

### BigVGAN reconstruction gate

The balanced held-out run used 30 clips per species (90 total). All BigVGAN
outputs were finite 65,536-sample waveforms and the maximum clipped fraction
was `0.0` (the acceptance limit is `0.001`). Mean paired multi-resolution
errors were:

| Decoder | Spectral convergence | Log-magnitude L1 |
|---|---:|---:|
| Exact-mel Griffin-Lim | 0.502969 | 0.670275 |
| BigVGAN | **0.444641** | **0.638315** |

This is an automatic decoder pass. The per-clip CSV and balanced listening
manifest are in the ignored `runs/bigvgan_real_test/` directory.

### Historical WGAN-GP pilot

The following is the original single-seed pilot retained as baseline evidence;
it is not the new canonical matrix selection. The 20-epoch CUDA run took
331.31 seconds. Selection minimizes
`abs(log(time_detail_ratio)) + abs(log(frequency_detail_ratio))` on validation;
epoch 10 was selected (`selection_error=0.065870`, time ratio `1.007071`,
frequency ratio `0.942873`).

| Epoch | Time ratio | Frequency ratio | Selection error |
|---:|---:|---:|---:|
| 1 | 0.282234 | 0.744327 | 1.560294 |
| 2 | 0.436615 | 0.700011 | 1.185364 |
| 3 | 0.409300 | 0.681075 | 1.277389 |
| 4 | 0.478862 | 0.853342 | 0.894937 |
| 5 | 0.819013 | 0.692572 | 0.566998 |
| 6 | 0.689920 | 0.744908 | 0.665674 |
| 7 | 0.522918 | 0.848700 | 0.812381 |
| 8 | 0.473004 | 0.836807 | 0.926813 |
| 9 | 0.612496 | 0.783654 | 0.734001 |
| 10 | **1.007071** | **0.942873** | **0.065870** |
| 11 | 0.725359 | 0.873887 | 0.455894 |
| 12 | 0.915155 | 0.971202 | 0.117882 |
| 13 | 0.822966 | 0.825728 | 0.386330 |
| 14 | 0.796916 | 0.833921 | 0.408624 |
| 15 | 0.769636 | 0.931947 | 0.332317 |
| 16 | 0.928028 | 1.040879 | 0.114759 |
| 17 | 0.773475 | 0.929563 | 0.329903 |
| 18 | 0.734861 | 0.771297 | 0.567755 |
| 19 | 0.807980 | 0.880532 | 0.340446 |
| 20 | 0.796277 | 1.008535 | 0.236307 |

### Generated-output integration evidence

The selected generator produced 8 samples per species (24 total), denormalized
them, and decoded them with the frozen pure-PyTorch BigVGAN path.

- Status: **integration pass**; 24/24 waveforms valid; 0 silent samples.
- Raw generated mel range: `[-11.482668, 1.670096]`.
- Mean scaled-mel saturation fraction: `0.0`.
- Mean pairwise scaled-mel L2 distance: `42.896523` (per species: Robin `37.319160`, Cardinal `31.458685`, Sparrow `41.722755`).
- Mean waveform RMS: `0.087831`; mean peak: `0.847155`.
- Maximum waveform clipped fraction: `0.000458` (0.0458%, below the 0.1% gate).

The generated WAVs and `generated_manifest.csv` are in the ignored
`runs/wgan_gp_bigvgan_audio/` directory for local listening.

## Generator retraining matrix

The new matrix keeps the original WGAN run as the current-settings baseline
and compares it with the stability candidate across seeds 42, 123, and 777.
It also ports the canonical Transformer to the 80x256 rectangular contract and
compares 16x16 with 8x16 patches across the same seeds. The full validation
rows, generated-output gates, temperature sweep, conditioning diagnostics,
and selected artifact hashes are in
[`../bigvgan_matrix_report.md`](../bigvgan_matrix_report.md).

| Family | Candidate | Mean validation metric | Median validation metric | Selected seed |
|---|---|---:|---:|---:|
| WGAN-GP | current (`1e-4`, `1e-4`, 2 critic steps) | 0.126761 selection error | 0.118256 | 123 within candidate |
| WGAN-GP | stability (`5e-5`, `1e-4`, 3 critic steps) | **0.095554** selection error | **0.077806** | **42** |
| Transformer | 16x16 patches | -1.039920 validation NLL | -1.039412 | 42 within candidate |
| Transformer | 8x16 patches | **-1.144653** validation NLL | **-1.143977** | **42** |

The canonical publication bundle therefore contains the stability WGAN seed 42
and the 8x16 Transformer seed 42. Both checkpoints embed the 22.05 kHz /
65,536-sample / 80x256 BigVGAN contract, training-only scaler, classes,
configuration, seed, epoch, and metrics. The bundle contains only those
checkpoints, configs, model cards, hashes, and a small balanced WAV set;
BigVGAN weights, resumable checkpoints, caches, bulk arrays, and bulk audio
remain excluded.

Balanced output gates passed for all recorded final sets: finite 80x256 mels,
finite 65,536-sample audio, zero silence, and clipping below 0.1%. Transformer
temperature 0.8 is the listening choice after the requested 0.4/0.6/0.8/1.0
sweep; temperature did not affect checkpoint selection.

The legacy residual CNN and CRNN were run only on BigVGAN-decoded WAVs. Their
target-label accuracies are reported separately as conditioning diagnostics,
not validation losses or perceived-realism scores.

## Verdict

The BigVGAN decoder passes the held-out reconstruction gate and improves both
recorded mean multi-resolution metrics over the exact-mel Griffin-Lim baseline.
The retrained stability WGAN and 8x16 Transformer both pass the shape, range,
finite, waveform-length, clipping, and diversity handoff checks. The evidence
supports `decoder_test` as the long-lived BigVGAN branch, not a claim of
perceptual realism or correct species conditioning. Use the balanced listening
manifests and keep the frozen-classifier results in their diagnostic role.
