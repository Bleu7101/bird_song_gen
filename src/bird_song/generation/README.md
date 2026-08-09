# Canonical WGAN-GP track

This package contains the canonical WGAN-GP spectrogram generator plus shared
evaluation and Griffin-Lim decoding helpers. It uses the existing classifier
representation: normalized `[1, 128, 128]` log-mel arrays from
`artifacts/spectrograms`. The classifier checkpoint and preprocessing are not
modified.

The other canonical generator is the continuous autoregressive Transformer in
`src/bird_song/transformer/`. VAE v1/v2/v3 notebook artifacts are preserved on
`main`. Diffusion models live on the separate `Diffusion`, `Difussion`, and
`diffusion_vincent` branches and are not part of this package.

The first VAE-v3/diffusion CRNN augmentation result is documented separately in
[`reports/crnn_synthetic_augmentation_2026-08-09`](../../../reports/crnn_synthetic_augmentation_2026-08-09/README.md).
It evaluates whether frozen generated spectrogram pools help a downstream
classifier; it does not add either model to this canonical WGAN-GP/Transformer
comparison or establish audio realism.

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
