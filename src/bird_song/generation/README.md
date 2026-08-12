# Historical WGAN-GP baseline

This package contains the historical WGAN-GP spectrogram generator plus shared
evaluation and Griffin-Lim decoding helpers. It uses the existing classifier
representation: normalized `[1, 128, 128]` log-mel arrays from
`artifacts/spectrograms`. The classifier checkpoint and preprocessing are not
modified.

The other retained baseline is the continuous autoregressive Transformer in
`src/bird_song/transformer/`. VAE v1/v2/v3 notebook artifacts are preserved on
`main`. Diffusion models live on the separate `Diffusion`, `Difussion`, and
`diffusion_vincent` branches and are not part of this package.

The inference-only three-seed checkpoint study is documented in
[`reports/generator_checkpoint_evaluation_2026-08-12`](../../../reports/generator_checkpoint_evaluation_2026-08-12/).
Use `scripts/generate_checkpoint_pool.py` for deterministic checkpoint-pool
operations and `scripts/evaluate_generator_checkpoints.py` for `audit`,
`evaluate`, or `package`. The bulk pools remain ignored; only bounded report
evidence is intended for version control. For the current refresh, all three
VAE-v3 pools were regenerated, while the existing audited DDIM pools were
reused and no diffusion spectrogram generation was run.

Generated-sample classification uses the selected CRNN only. The report keeps
target-label accuracy, macro F1, Fréchet distance, feature precision and
recall, density, coverage, diversity, copy-risk diagnostics, and seed
stability. These are classifier-view and feature-space diagnostics, not a
claim of perceptual or waveform realism.

Across generation seeds 42, 123, and 777, VAE-v3 recorded 96.67% target-label
accuracy and 96.66% macro F1; Diffusion recorded 92.44% and 92.42%. The VAE
checkpoint was not retrained. Its existing posterior bank was filtered to
256 Northern Cardinal, 247 Song Sparrow, and 256 American Robin anchors before
the VAE pools were regenerated. The comparison uses the 510-row generator-safe
validation manifest for copy-risk thresholds and the unchanged 489-row test
manifest for CRNN calibration.

Use `scripts/generate_checkpoint_pool.py --benchmark` to benchmark fresh
in-memory sampling and `scripts/package_generator_speed.py` to produce the
matched VAE-v3/DDIM speed report. The timing boundary excludes model loading,
warm-up, classifier-scale conversion, CPU transfer, array serialization,
plotting, and audio decoding.

The recorded FP32 CUDA benchmark generated 600 spectrograms per repeat at
batch size 8. VAE-v3 averaged 0.3058 seconds and Diffusion's 100-step DDIM
sampler averaged 412.3090 seconds, a 1348.09x Diffusion-to-VAE time ratio on
the recorded RTX 4070 SUPER. The completed diffusion benchmark was retained,
and VAE-v3 was remeasured with the filtered 759-anchor bank; no diffusion
sampling was rerun. Runtime is separate from quality and downstream utility.

Downstream utility is reported in
[`reports/crnn_low_resource_augmentation_2026-08-12`](../../../reports/crnn_low_resource_augmentation_2026-08-12/).
The preserved nine-block study selected +200/species for both generators. Mean
held-out macro F1 was 84.75% for the newly initialized real-only CRNNs, 87.48%
with VAE-v3 (+2.72 percentage points; 7/9 positive blocks; descriptive interval
+1.00 to +4.65), and 86.21% with Diffusion (+1.46 points; 6/9; +0.15 to
+2.89). This is a simulated low-resource result, not a comparison against the
historical full-data CRNN.

## Train and sample

All scripts use `--device auto`, so CUDA is selected when available. Use
`--device cpu` to force a CPU run. The smoke flags below limit work to one
batch and one epoch; they verify wiring but are not model results.

```powershell
$env:PYTHONPATH = "src"
$py = "..\bird_song_venv\Scripts\python.exe"

& $py scripts/08_train_wgan_gp.py --device auto
& $py scripts/08_generate_wgan_gp.py --checkpoint runs/wgan_gp/best.pt
```

For a wiring smoke test, add `--epochs 1 --max-batches 1 --batch-size 2
--workers 0 --overwrite` and use a separate output directory.

## Evaluation

Compare generated manifests with the same validation split:

```powershell
& $py scripts/11_evaluate_generators.py `
    --generated-manifest outputs/wgan_gp/generated_manifest.csv
```

The evaluation reports finite/range checks, time/frequency detail ratios, and
pairwise diversity. Classifier agreement remains a secondary diagnostic; it is
not treated as a realism score.

Decode any generated manifest to WAV with the same fixed Griffin-Lim baseline:

```powershell
& $py scripts/12_decode_generated_audio.py `
    --manifest outputs/wgan_gp/generated_manifest.csv `
    --output-dir outputs/wgan_gp_audio
```

This decoder is a controlled comparison, not a claim that Griffin-Lim restores
the phase information discarded by the mel transform.

Future work comparing alternative methods for converting generated
spectrograms into audio waveforms belongs on the `BigVGAN_decode` branch. That
branch currently uses a raw `80 x 256` BigVGAN-compatible mel contract, whereas
this package uses normalized `128 x 128` mels. Do not exchange cached mels or
generator checkpoints between those contracts without an explicit conversion
and validation step.
