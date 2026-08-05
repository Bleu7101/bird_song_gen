# Generator tracks

These models share the existing classifier representation: normalized
`[1, 128, 128]` log-mel arrays from `artifacts/spectrograms`. The classifier
checkpoint and preprocessing are not modified.

## Recommended order

1. Train/sample WGAN-GP as the direct sharpness experiment.
2. Train VQGAN as a sharp tokenizer/decoder, then train the token transformer.
3. Train latent diffusion only after the VQGAN representation is useful.

All scripts use `--device auto`, so CUDA is selected when available. Use
`--device cpu` to force a CPU run. The smoke flags below limit work to one
batch and one epoch; they verify wiring but are not model results.

```powershell
$env:PYTHONPATH = "src"
$py = "..\bird_song_venv\Scripts\python.exe"

& $py scripts/08_train_wgan_gp.py --device auto
& $py scripts/08_generate_wgan_gp.py --checkpoint runs/wgan_gp/best.pt

& $py scripts/09_train_vqgan.py --device auto
& $py scripts/09_train_token_transformer.py `
    --vqgan-checkpoint runs/vqgan/best.pt --device auto
& $py scripts/09_generate_token_transformer.py `
    --checkpoint runs/token_transformer/best.pt `
    --vqgan-checkpoint runs/vqgan/best.pt

& $py scripts/10_train_latent_diffusion.py `
    --vqgan-checkpoint runs/vqgan/best.pt --device auto
& $py scripts/10_generate_latent_diffusion.py `
    --checkpoint runs/latent_diffusion/best.pt `
    --vqgan-checkpoint runs/vqgan/best.pt
```

For a wiring smoke test, add `--epochs 1 --max-batches 1 --batch-size 2
--workers 0 --overwrite` and use a separate output directory.

## Evaluation

Compare generated manifests with the same validation split:

```powershell
& $py scripts/11_evaluate_generators.py `
    --generated-manifest outputs/wgan_gp/generated_manifest.csv `
    --generated-manifest outputs/token_transformer/generated_manifest.csv `
    --generated-manifest outputs/latent_diffusion/generated_manifest.csv
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
