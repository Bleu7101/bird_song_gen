# Bird Song Generation

This repository studies conditional generation of three-second bird-song
log-mel spectrograms for American Robin, Northern Cardinal, and Song Sparrow.
The shared representation is a normalized `1 x 128 x 128` log-mel tensor.

## Recorded results at a glance

The selected CRNN reached **90.16% held-out macro F1** on real audio. Generator
evaluation now uses that CRNN only and reports target-label accuracy, macro F1,
Fréchet distance, feature precision, feature recall, density, coverage,
diversity, copying-risk diagnostics, and generation-seed stability. The
low-resource augmentation study retains exactly nine matched blocks and seven
conditions per block: `50+0`, VAE-v3 `50+50/100/200`, and diffusion
`50+50/100/200`. The completed study selected +200/species for both generators.
Mean held-out macro F1 was **84.75%** for real-only, **87.48%** for VAE-v3,
and **86.21%** for Diffusion. The paired changes were **+2.72 percentage
points** for VAE-v3 (7/9 positive blocks; descriptive interval +1.00 to +4.65)
and **+1.46 points** for Diffusion (6/9 positive blocks; +0.15 to +2.89).
These results apply only to newly initialized low-resource CRNNs, not the
historical full-data CRNN.

Across the evaluated three-seed pools, VAE-v3 reached **96.67% target-label
accuracy** and **96.66% macro F1**, while Diffusion reached **92.44%** and
**92.42%**.
These are selected-CRNN compatibility metrics, not realism scores. In the
matched generator-only benchmark, VAE-v3 averaged **0.3058 s** and Diffusion
**412.3090 s** per 600 spectrograms on the same RTX 4070 SUPER; Diffusion took
**1348.09x** the VAE generator time. The timing excludes loading, transfers,
disk I/O, and waveform decoding.

The VAE checkpoint was **not retrained**. Its existing posterior bank was
filtered to 256 Northern Cardinal, 247 Song Sparrow, and 256 American Robin
anchors, then the three VAE pools were regenerated with
`z = mu + 0.35 * std * epsilon`. The already-audited diffusion pools were
reused and no diffusion spectrogram generation was run. Generator comparisons
use the 510-row generator-safe validation manifest and the unchanged 489-row
held-out test manifest.

## Main branch overview

`main` is the primary review branch. It contains the shared data pipeline, the
classifier study, VAE v1/v2/v3 artifacts, and two historical spectrogram
generation baselines: the continuous Transformer and WGAN-GP. Decoder research
and diffusion implementations remain isolated on separate branches so their
different model and representation contracts are not mixed into the main
workflow. The maintained evaluation uses three classifier-ready pools per
model, scores them with the selected CRNN, compares generator-only sampling
speed, and feeds those same pools into the matched low-resource study. The VAE
pools were regenerated after the sampling correction; the audited diffusion
pools were reused.

## Branch guide

| Branch | Purpose |
|---|---|
| [`main`](https://github.com/Bleu7101/bird_song_gen/tree/main) | Primary pipeline, classifier evidence, VAE v1/v2/v3 experiments, and historical Transformer and WGAN-GP generator baselines |
| [`BigVGAN_decode`](https://github.com/Bleu7101/bird_song_gen/tree/BigVGAN_decode) | Future work for testing alternative decoders/vocoders that convert generated spectrograms into audio waveforms; the current BigVGAN results are the first recorded decoder experiment |
| [`Diffusion`](https://github.com/Bleu7101/bird_song_gen/tree/Diffusion), [`Difussion`](https://github.com/Bleu7101/bird_song_gen/tree/Difussion), and [`diffusion_vincent`](https://github.com/Bleu7101/bird_song_gen/tree/diffusion_vincent) | Separate branches containing diffusion-model work; use the documentation, configs, and artifacts on the relevant branch |

Branch names identify isolated workstreams, not additional stages of the
`main` pipeline. In particular, the decoder branch changes the mel and waveform
contract, and diffusion code should be evaluated from its own branch.

| Area | Implementation | Evidence/status |
|---|---|---|
| Dataset and splits | `scripts/01_create_splits.py` and `manifests/` | Historical v1: 2,339 train, 519 validation, 489 test; `content_safe_v2` has 2,315 train, while maintained generator comparisons use a 510-row generator-safe validation subset and the unchanged 489-row test set |
| Shared preprocessing | `scripts/02_build_spectrograms.py` and `configs/spectrogram.json` | Manifest-backed normalized spectrogram cache under local `artifacts/spectrograms/` |
| Classifier | `src/bird_song/classifier/` and `scripts/03_*.py` | Four architectures, three seeds each; CRNN selected from validation evidence |
| Low-resource CRNN augmentation | `scripts/15_crnn_low_resource_augmentation.py` and `reports/crnn_low_resource_augmentation_2026-08-12/` | Completed 63-run, nine-block matched study; +200/species selected for both generators, with mean macro-F1 changes of +2.72 points for VAE-v3 and +1.46 points for Diffusion versus matched real-only controls |
| Generator checkpoint evaluation | `scripts/evaluate_generator_checkpoints.py` and `reports/generator_checkpoint_evaluation_2026-08-12/` | CRNN-only target compatibility, feature-distribution, diversity, copying-risk, and seed-stability diagnostics |
| Generator-only speed | `scripts/generate_checkpoint_pool.py --benchmark` and `scripts/package_generator_speed.py` | Same-device VAE-v3 versus DDIM sampling time; excludes loading, disk I/O, plotting, and audio decoding |
| Historical Transformer baseline | `src/bird_song/transformer/` and `scripts/06_*.py` | Fully trained working baseline with a documented caveated verdict; run files are archived under `reports/generator_baselines/transformer/` |
| Historical WGAN-GP baseline | `src/bird_song/generation/wgan_gp.py` and `scripts/08_*.py` | Full-split 20-epoch run with recorded evidence and curated audio under `reports/generator_baselines/wgan_gp/`; bulk runs remain local |
| VAE v1/v2/v3 | `notebooks/04_conditional_vae.ipynb`, `artifacts/models/vae/`, and `reports/vae/` | Three recorded VAE versions are preserved on `main`; the portable V3 pool sampler applies temperature in reparameterization and the checkpoint was not retrained |
| Diffusion models | Separate diffusion branches listed above | Training implementations remain outside `main`; evaluation uses the external validation-best epoch-34 EMA checkpoint with the final DDIM notebook settings |

## Decoder roadmap

The historical generators on `main` produce normalized `128 x 128` log-mel
spectrograms, and Griffin-Lim remains the fixed baseline used by the recorded
main-branch audio demonstrations. Future comparisons of different methods for
turning generated representations into audio waveforms belong on
`BigVGAN_decode`.

That branch currently records a BigVGAN-compatible `80 x 256` experiment. Its
representation is intentionally different from the `128 x 128` contract on
`main`, so checkpoints, cached mels, and decoder metrics must not be mixed
across the two branches. Decoder experiments are waveform-rendering studies,
not extra generator families.

## Repository roles

- `src/bird_song/` is the importable source of truth.
- `scripts/` contains thin command-line entry points for reproducible runs.
- `notebooks/` contains visual, gated companions that import the `src` code;
  notebooks are not a second classifier implementation.
- `notebooks/03_classifier.ipynb` is the CRNN-focused classifier companion.
- `notebooks/06_evaluation.ipynb` is the primary four-part evaluation report:
  generator quality and seed stability, low-resource augmentation,
  generator-only speed, and the evidence-bounded conclusion.
- `notebooks/generator_checkpoint_evaluation.ipynb` and
  `notebooks/low_resource_crnn_augmentation.ipynb` remain focused report-only
  companions; neither requires ignored generator checkpoints or pools.
- `notebooks/autoregressive_transformer.ipynb` and `notebooks/wgan_gp.ipynb`
  are unnumbered extra/future-work companions for the historical generator
  baselines.
- The merged legacy preprocessing notebook is retained as
  `notebooks/02_preprocess_logmel_legacy_processed.ipynb`; the canonical
  preprocessing companion is `notebooks/02_preprocess_logmel.ipynb`.
- `artifacts/models/classifier/` and `artifacts/models/vae/` contain versioned
  model packages and reusable banks. Their READMEs explain the
  validation-versus-held-out boundaries.
- `reports/` contains curated evidence. `runs/` and `outputs/` are local,
  ignored workspaces for fresh training and generated samples.

## Classifier: selection and recorded results

The controlled comparison used the same recording-ID-isolated historical v1
splits, preprocessing, optimizer, width, dropout, early-stopping rule, and seeds
for every architecture.

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
90.39% accuracy and 90.44% macro F1. It is also retained as the frozen training
teacher recorded by VAE-v3. These are real-audio held-out metrics; maintained
generated-sample scoring uses the selected CRNN only.

The selected model package is [`artifacts/models/classifier/selected_crnn`](artifacts/models/classifier/selected_crnn/README.md).
The complete sweep is documented in
[`artifacts/models/classifier/architecture_comparison`](artifacts/models/classifier/architecture_comparison/README.md),
and the legacy checkpoint is described in
[`artifacts/models/classifier/Harvey_classifier`](artifacts/models/classifier/Harvey_classifier/README.md).

Evaluate the selected checkpoint on a new labeled manifest with:

```powershell
$env:PYTHONPATH = "src"
& $py scripts/03_evaluate_classifier.py `
  --checkpoint artifacts/models/classifier/selected_crnn/best.pt `
  --output-dir runs/selected_crnn_test `
  --workers 0 --device auto
```

For architecture comparison, use the controlled sweep command in the
architecture-comparison README. Do not select architectures by repeatedly
checking the held-out test split.

## Low-resource CRNN augmentation experiment

The matched low-resource study asks a narrow question: if only 50 labeled real
spectrograms per species are visible to a newly initialized CRNN, can the
audited generated spectrogram pools improve classification of real held-out
clips? Each real row comes from a distinct recording ID. Three deterministic
real subsets
are crossed with three CRNN initialization seeds, while generation-pool seeds
42, 123, and 777 rotate across the resulting nine matched blocks. The sweep
trains seven conditions per block: real-only and VAE-v3/diffusion additions of
50, 100, or 200 per species, for 63 training runs overall.

The exact matched conditions are:

- `50+0`: 50 real spectrograms per species and no generated data.
- VAE-v3 `50+50`, `50+100`, and `50+200` per species.
- Diffusion `50+50`, `50+100`, and `50+200` per species.

The 510-row generator-safe validation split selects one ratio independently
for each generator before the matched real-only and selected generator arms
are evaluated on the unchanged 489-row test split. The final report contains
all 63 validation runs and exactly 27 test rows: nine
matched blocks for real-only plus each selected generator arm. Report accuracy,
macro F1, paired deltas, positive-block counts, sample variation, and a
descriptive block-bootstrap interval. These are classifier-utility results,
not perceptual-realism results.

Validation selected +200 generated spectrograms per species for both models,
the largest tested ratio. On held-out real clips, mean macro F1 increased from
84.75% for real-only to 87.48% with VAE-v3 and 86.21% with Diffusion. The mean
paired changes were +2.72 points (7/9 positive blocks; descriptive interval
+1.00 to +4.65) and +1.46 points (6/9; +0.15 to +2.89), respectively. Because
the real subsets overlap and there are only nine matched blocks, these are
descriptive matched-study intervals, not p-values or independent-replication
confidence intervals.

The design simulates label scarcity for three common project species rather
than genuinely rare or unseen species, and the pretrained generators had
access to more source data than the low-resource CRNN. The three real subsets
also overlap because the training split contains only 56 Robin, 63 Cardinal,
and 95 Sparrow recording IDs, so the nine blocks are not nine independent
datasets. A selected ratio is the best tested value among `50/100/200`, not an
estimated optimum.

The portable evidence is packaged in
[`reports/crnn_low_resource_augmentation_2026-08-12`](reports/crnn_low_resource_augmentation_2026-08-12/),
with a visual companion in
[`notebooks/low_resource_crnn_augmentation.ipynb`](notebooks/low_resource_crnn_augmentation.ipynb).
Bulk checkpoints and histories stay under ignored
`runs/crnn_low_resource_augmentation/v3/`.

Audit the regenerated VAE pools, reused diffusion pools, and subset design, or
resume the full experiment, with:

```powershell
& $py scripts/15_crnn_low_resource_augmentation.py audit
& $py scripts/15_crnn_low_resource_augmentation.py run --device cuda --workers 0
```

The command never trains or samples a generator. It refuses incompatible
existing CRNN runs and resumes only checkpoints whose complete protocol
signature matches the requested experiment.

## Historical generation baselines and evaluation boundary

The repository retains exactly two generator baselines for historical
comparison: the continuous autoregressive Transformer and WGAN-GP. Shared
evaluation and Griffin-Lim decoding helpers support those models but are not
additional generators. The corrected VAE-v3/diffusion checkpoint study and its
matched low-resource classifier-utility experiment are the current evaluation
focus.

The inference-only three-seed checkpoint study is documented in
[`reports/generator_checkpoint_evaluation_2026-08-12`](reports/generator_checkpoint_evaluation_2026-08-12/)
and its report-only companion is
[`notebooks/generator_checkpoint_evaluation.ipynb`](notebooks/generator_checkpoint_evaluation.ipynb).
The seed-42/123/777 VAE pools were regenerated under ignored
`runs/generator_checkpoint_evaluation/pools/vae_v3/`; the corresponding
diffusion pools were audited and reused without new diffusion sampling. The
selected CRNN is the sole generated-sample classifier. The report retains
target-label accuracy, macro F1, Fréchet distance, feature precision and
recall, density, coverage, diversity, nearest-neighbor copying-risk
diagnostics, and seed stability. These remain classifier-view and
feature-space diagnostics, not perceptual-realism scores.

Generator-only sampling speed is packaged in
[`reports/generator_speed_comparison_2026-08-12`](reports/generator_speed_comparison_2026-08-12/).
Each recorded repeat generated 200 spectrograms per species with batch size 8
on the same device. Timed repeats include only tensor sampling; model loading,
warm-up, disk writes, plotting, and audio decoding are outside the timing
boundary.

The continuous autoregressive transformer is documented in
[`reports/generator_baselines/transformer/README.md`](reports/generator_baselines/transformer/README.md).
Its best tested temperature-1.0 run achieved 39.58% target-label agreement with
the selected CRNN.
Classifier agreement is a diagnostic for generated samples, not proof of
acoustic realism; the model card records the visual and conditioning caveats.

The full-split WGAN-GP run is summarized in
[`reports/generator_baselines/wgan_gp/README.md`](reports/generator_baselines/wgan_gp/README.md).
That report records selected-CRNN agreement (40.63%), detail ratios, listening,
and visual inspection.

The earlier VQGAN, token-Transformer, and latent-diffusion wiring pilots are
not maintained on `main` because they were short integration checks, not
comparable generator experiments. They are distinct from the dedicated
diffusion-model work on the separate diffusion branches. Existing ignored
local checkpoints and outputs are left untouched, while the recorded VAE
v1/v2/v3 artifacts remain on `main`.

Generated-sample evaluation is run with species-named parent directories:

```powershell
$env:PYTHONPATH = "src"
& $py scripts/07_evaluate_generated.py `
  --checkpoint artifacts/models/classifier/selected_crnn/best.pt `
  --input generated_samples --labels-from-parent
```

The classifier is closed-set and cannot reject noise or unknown species, so
generated-audio conclusions must include listening and spectrogram review.

## Setup and smoke test

The dataset is intentionally not committed. Place the supplied dataset at
`bird_songs_dataset/` with `wavfiles/` and `bird_songs_metadata.csv`, then run
from the repository root in PowerShell:

```powershell
$py = "..\bird_song_venv\Scripts\python.exe"
$env:PYTHONPATH = (Resolve-Path "src").Path
& $py -m pip install -r requirements.txt
& $py -m pip install -e .
& $py scripts/02_build_spectrograms.py --output-dir artifacts/spectrograms
& $py -m pytest
& $py scripts/03_train_classifier.py --architecture crnn --dry-run --workers 0 --device auto
```

The smoke test loads one batch, checks the `(batch, 1, 128, 128)` input shape,
and performs no optimizer step. Training and evaluation commands refuse to
silently overwrite existing run files. Classifier commands default to the
manifest-backed cache at `artifacts/spectrograms/`; pass `--spectrogram-cache`
to use another validated cache.

## Reproducible pipeline commands

```powershell
& $py scripts/01_create_splits.py --dry-run
& $py scripts/02_build_spectrograms.py --limit 8 --output-dir artifacts/spectrograms_smoke
& $py scripts/03_train_classifier.py --architecture crnn --epochs 40 --batch-size 64 --workers 4 --device cuda
& $py scripts/06_train_transformer.py --epochs 60 --batch-size 32 --workers 4 --device cuda
& $py scripts/08_train_wgan_gp.py --epochs 20 --critic-steps 2 --batch-size 32 --workers 0 --device cuda
```

The VAE v1/v2/v3 notebook, checkpoints, and recorded outputs are preserved on
`main`. Diffusion models are maintained on the separate diffusion branches
listed in the branch guide. Any cross-model comparison should first align the
preprocessing contract, checkpoint format, evaluation split, and audio-decoder
path.
