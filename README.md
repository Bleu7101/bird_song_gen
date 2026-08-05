# Bird Song Generation

This repository studies conditional generation of three-second bird-song
log-mel spectrograms for American Robin, Northern Cardinal, and Song Sparrow.
The shared representation is a normalized `1 x 128 x 128` log-mel tensor.

## Harvey_classifier review snapshot

This branch contains Harvey's classifier work, the continuous transformer
baseline, and the later generator pilots. The branch is intentionally not a
merge of the separate `Diffusion` branch; that branch remains notebook-based
and is being developed independently.

| Area | Canonical implementation | Evidence/status |
|---|---|---|
| Dataset and splits | `scripts/01_create_splits.py` and `manifests/` | 2,339 train, 519 validation, 489 test clips; recording-safe splits |
| Shared preprocessing | `scripts/02_build_spectrograms.py` and `configs/spectrogram.json` | Reproducible cache under local `artifacts/spectrograms/` |
| Classifier | `src/bird_song/classifier/` and `scripts/03_*.py` | Four architectures, three seeds each; CRNN selected from validation evidence |
| VAE | `notebooks/04_conditional_vae.ipynb` | Notebook experiment and recorded outputs; no claim of a reusable `src` implementation on this branch |
| Diffusion | `Diffusion` branch notebook | Not merged here; its checkpoint is not part of this branch's reproducible artifacts |
| Transformer | `src/bird_song/transformer/` and `scripts/06_*.py` | Working baseline with a documented caveated verdict |
| Generator pilots | `src/bird_song/generation/` and `scripts/08_*.py` through `scripts/12_*.py` | WGAN, VQGAN/token, latent-diffusion wiring and recorded pilot evidence |

## Setup and smoke test

The dataset is intentionally not committed. Place the supplied dataset at
`bird_songs_dataset/` with `wavfiles/` and `bird_songs_metadata.csv`, then run
from the repository root in PowerShell:

```powershell
$py = "..\bird_song_venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pip install -e .
& $py -m pytest
& $py scripts/03_train_classifier.py --architecture crnn --dry-run --workers 0 --device auto
```

The smoke test loads one batch, checks the `(batch, 1, 128, 128)` input shape,
and performs no optimizer step. Training and evaluation commands refuse to
silently overwrite existing run files.

## Repository roles

- `src/bird_song/` is the importable source of truth.
- `scripts/` contains thin command-line entry points for reproducible runs.
- `notebooks/` contains visual, gated companions that import the `src` code;
  notebooks are not a second classifier implementation.
- The merged legacy preprocessing notebook is retained as
  `notebooks/02_preprocess_logmel_legacy_processed.ipynb`; the canonical
  preprocessing companion is `notebooks/02_preprocess_logmel.ipynb`.
- `classifier_artifacts/` contains versioned model weights and evaluation
  evidence. Its index explains the validation-versus-held-out boundary.
- `reports/` contains curated generator evidence. Bulk generated arrays remain
  local and are ignored.

## Classifier: selection and recorded results

The controlled comparison used the same recording-safe splits, preprocessing,
optimizer, width, dropout, early-stopping rule, and seeds for every architecture.

| Architecture | Parameters | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|
| CRNN (CNN-GRU) | 404,451 | **90.56% +/- 1.39%** | **90.45% +/- 1.45%** |
| Plain CNN | 1,238,691 | 90.43% +/- 1.74% | 90.33% +/- 1.76% |
| Residual CNN | 1,661,795 | 87.54% +/- 0.99% | 87.44% +/- 0.97% |
| Depthwise CNN | 137,699 | 81.63% +/- 2.70% | 81.67% +/- 2.94% |

The CRNN seed-777 checkpoint was preselected because it had the strongest
selected-run validation result: 92.10% accuracy and 92.06% macro F1 at epoch
19. It was then evaluated once on the held-out test split:

| Metric | CRNN selected checkpoint |
|---|---:|
| Accuracy | **89.98%** |
| Macro precision | 90.25% |
| Macro recall | 90.18% |
| Macro F1 | **90.16%** |

The earlier residual CNN remains available as a legacy held-out baseline with
90.39% accuracy and 90.44% macro F1. It is retained because the earlier
transformer and generator reports explicitly used that checkpoint. These are
real-audio held-out metrics; they are not generated-sample realism scores.

The selected model package is [`classifier_artifacts/selected_crnn`](classifier_artifacts/selected_crnn/README.md).
The complete sweep is documented in
[`classifier_artifacts/architecture_comparison`](classifier_artifacts/architecture_comparison/README.md),
and the legacy checkpoint is described in
[`classifier_artifacts/Harvey_classifier`](classifier_artifacts/Harvey_classifier/README.md).

Evaluate the selected checkpoint on a new labeled manifest with:

```powershell
$env:PYTHONPATH = "src"
& $py scripts/03_evaluate_classifier.py `
  --checkpoint classifier_artifacts/selected_crnn/best.pt `
  --output-dir runs/selected_crnn_test `
  --workers 0 --device auto
```

For architecture comparison, use the controlled sweep command in the
architecture-comparison README. Do not select architectures by repeatedly
checking the held-out test split.

## Generation baselines and evaluation boundary

The continuous autoregressive transformer is documented in
[`runs/transformer_generator/README.md`](runs/transformer_generator/README.md).
Its best tested temperature-1.0 run achieved 80.73% target-label agreement
with the legacy residual classifier but only 39.58% with the independent CRNN.
Classifier agreement is a diagnostic for generated samples, not proof of
acoustic realism; the model card records the visual and conditioning caveats.

The WGAN-GP pilot and quick VQGAN/token/diffusion pilots are summarized in
[`reports/generator_pilot_2026-08-04/README.md`](reports/generator_pilot_2026-08-04/README.md).
That report likewise separates residual-classifier agreement (84.90%) from
CRNN agreement (40.63%), detail ratios, listening, and visual inspection.

Generated-sample evaluation is run with species-named parent directories:

```powershell
$env:PYTHONPATH = "src"
& $py scripts/07_evaluate_generated.py `
  --checkpoint classifier_artifacts/selected_crnn/best.pt `
  --input generated_samples --labels-from-parent
```

The classifier is closed-set and cannot reject noise or unknown species, so
generated-audio conclusions must include listening and spectrogram review.

## Reproducible pipeline commands

```powershell
& $py scripts/01_create_splits.py --dry-run
& $py scripts/02_build_spectrograms.py --limit 8 --output-dir artifacts/spectrograms_smoke
& $py scripts/03_train_classifier.py --architecture crnn --epochs 40 --batch-size 64 --workers 4 --device cuda
& $py scripts/06_train_transformer.py --epochs 60 --batch-size 32 --workers 4 --device cuda
```

The VAE notebook and the separate Diffusion notebook should be treated as
interactive experiments until their owners publish portable source,
checkpoints, and evidence on their respective branches.
