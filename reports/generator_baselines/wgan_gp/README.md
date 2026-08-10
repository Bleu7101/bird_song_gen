# WGAN-GP pilot - 2026-08-04

This retained report covers the historical WGAN-GP baseline only. Bulk training
outputs and checkpoints remain in the ignored local `runs/` and `outputs/`
workspaces.

## Outcome

The conditional WGAN-GP is the branch's first useful sharp-spectrogram
baseline. It trained for 20 epochs over the full cached training split on CUDA
and produced 64 samples per species. One listening sample is encouraging, but
the evidence does not support an audio-ready claim.

The frozen residual classifier matched the requested class on 84.90% of 192
WGAN samples. The independent CRNN matched only 40.63%, heavily favoring
American Robin. This disagreement is the main reason not to treat classifier
readability as realism.

Generated time/frequency detail ratios were:

| Species | Time | Frequency |
|---|---:|---:|
| American Robin | 1.67 | 1.67 |
| Northern Cardinal | 1.49 | 1.90 |
| Song Sparrow | 0.69 | 1.11 |

Values near 1.0 match real validation detail energy. The overshoot for Robin
and Cardinal may be sharp structure, noise, or both. Listening and visual
inspection therefore remain necessary.

## Curated audio

- [American Robin](audio/american_robin.wav)
- [Northern Cardinal](audio/northern_cardinal.wav)
- [Song Sparrow](audio/song_sparrow.wav)

These are three-second Griffin-Lim demonstrations, peak-normalized to 0.95.
The decoder cannot recover phase discarded by the log-mel representation.

Griffin-Lim remains the fixed decoder baseline for this historical report.
Future work comparing other ways to convert generated spectrograms into audio
waveforms is isolated on the `BigVGAN_decode` branch so decoder changes do not
alter the evidence recorded here.

## Reproduction

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
$py = "..\bird_song_venv\Scripts\python.exe"

& $py scripts/08_train_wgan_gp.py `
    --epochs 20 --critic-steps 2 --batch-size 32 --workers 0 `
    --device auto --output-dir runs/wgan_gp_real --overwrite

& $py scripts/08_generate_wgan_gp.py `
    --checkpoint runs/wgan_gp_real/best.pt `
    --samples-per-species 64 --output-dir outputs/wgan_gp_real `
    --device auto --overwrite
```

The large checkpoints and bulk arrays remain local and ignored. The WGAN
history file in this directory is the recorded training evidence.

## Later downstream evidence (2026-08-09)

The first CRNN synthetic-augmentation evaluation later tested frozen VAE-v3
and diffusion pools; it is recorded in
[`crnn_synthetic_augmentation_2026-08-09`](../../crnn_synthetic_augmentation_2026-08-09/README.md).
It did not evaluate WGAN-GP and therefore does not revise this report's
historical WGAN verdict.

## Next experiment

Train WGAN-GP v2 with limited-data discriminator augmentation and three seeds.
Change checkpoint selection so it rewards detail ratios close to 1.0,
diversity, and cross-classifier consistency; the current maximum-detail rule
can select over-sharp noisy checkpoints.
