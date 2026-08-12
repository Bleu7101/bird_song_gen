# Three-seed checkpoint generator evaluation

## Results at a glance

This is an inference-only, classifier-view comparison of frozen VAE-v3 and
conditional-diffusion checkpoints. Each model contributes three deterministic
pools (seeds 42, 123, and 777), with 200 spectrograms per species per seed:
1,800 generated arrays per model and 3,600 total. Only the selected CRNN is
used as the evaluator.

| Generator | Target-label accuracy | Macro F1 | Seed range, accuracy |
|---|---:|---:|---:|
| VAE-v3 | 96.67% ± 1.20 pp | 96.66% ± 1.21 pp | 95.67%–98.00% |
| Diffusion | 92.44% ± 0.86 pp | 92.42% ± 0.84 pp | 91.50%–93.17% |

Values are the mean and sample standard deviation over the three generation
seeds. They measure whether the frozen CRNN recognizes the requested species;
they are not direct measures of audio realism.

### Target-label accuracy by species

| Generator | American Robin | Northern Cardinal | Song Sparrow |
|---|---:|---:|---:|
| VAE-v3 | 99.67% ± 0.29 pp | 96.67% ± 1.53 pp | 93.67% ± 2.47 pp |
| Diffusion | 94.00% ± 0.50 pp | 84.33% ± 1.76 pp | 99.00% ± 1.00 pp |

### Classifier-feature and image diagnostics

The following values are macro-averages over species after first aggregating
over the three generation seeds. Fréchet distance and pixel Wasserstein
distance are better when lower; feature precision, feature recall, density,
and coverage are better when higher. A diversity ratio near 1 means generated
feature dispersion is close to real-test dispersion.

| Generator | Fréchet distance | Feature precision | Feature recall | Density | Coverage | Diversity ratio | Pixel Wasserstein |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE-v3 | 22.8950 | 0.8528 | 0.8972 | 0.6423 | 0.7419 | 0.8719 | 0.0787 |
| Diffusion | 41.2550 | 0.5654 | 0.8207 | 0.3114 | 0.4845 | 1.0083 | 0.1916 |

In this frozen-CRNN representation, VAE-v3 has stronger conditioning and
closer real-test feature and pixel distributions. Diffusion's diversity ratio
is closer to 1. These observations do not establish a perceptual-quality
winner.

### Copy-risk screen

The screening threshold for each species is the fifth percentile of
generator-safe validation-to-training nearest-neighbor distance. VAE-v3 has
660 of 1,800 generated arrays (36.67%) at or below the relevant threshold;
diffusion has 45 of 1,800 (2.50%). Across species and seeds, the corresponding
fractions range from 16.5% to 50.5% for VAE-v3 and from 0.0% to 6.5% for
diffusion. No generated array is an exact match to a training array, and each
600-array pool contains 600 unique arrays.

A below-threshold result is a conservative similarity flag, not proof that a
sample was copied. The higher VAE-v3 rate is therefore a limitation to carry
alongside its stronger classifier-view distribution metrics.

## Refresh boundary

The VAE checkpoint was **not retrained**. Its existing posterior bank was
filtered against `content_safe_v2/full_dataset_train.csv`: Northern Cardinal
retained 256 anchors, Song Sparrow retained 247 of 256 anchors, and American
Robin retained 256 anchors. The nine removed Song Sparrow anchors were absent
from the current content-safe training manifest. The refreshed VAE pools use
the corrected reparameterization
`z = mu + 0.35 * exp(0.5 * logvar) * epsilon`.

The existing diffusion pools were reused for this refreshed evaluation; no
diffusion spectrogram generation was run. Their recorded contract is the
epoch-34 validation-best EMA checkpoint with DDIM, 100 sampling steps,
`eta=0`, guidance 3.0, and clamp 4.0. The checkpoint remains external under
`C:\Users\Harvey\Desktop\conditional_diffusion` and is not copied into this
report.

The copy-risk threshold now uses the 510-row generator-safe validation
manifest. It is a strict subset of the earlier 519-row validation manifest,
excluding nine Song Sparrow rows identified by the existing exact-duplicate
ledger as retained counterparts of historical-training rows. The held-out test
manifest is unchanged at 489 rows (145 American Robin, 162 Northern Cardinal,
and 182 Song Sparrow). The nine validation exclusions and the nine VAE-bank
anchor removals are separate filters and should not be treated as the same
records.

The selected CRNN reproduces 89.98% accuracy and 90.16% macro F1 on that
unchanged real test set. Feature standardization and 64-component PCA are fit
only on the content-safe training split. Sample-sensitive feature metrics use
200 deterministic resamples of 128 real and 128 generated examples per
species.

## Evidence files

- `classifier_scores.csv` contains one conditioning row per model and seed.
- `metrics_per_seed.csv` contains all 18 model/seed/species rows.
- `metrics_aggregate.csv` contains mean, sample standard deviation, minimum,
  and maximum over the three seeds for every recorded metric.
- `nearest_neighbor_summary.csv` contains the copy-risk threshold and counts.
- `confusion_matrices/` contains the six generated-pool matrices and the
  unchanged real-test calibration matrix.
- `figures/conditioning_by_seed.png` and
  `figures/feature_distance_summary.png` visualize conditioning stability and
  feature distance.
- `protocol.json`, `provenance.json`, `pool_audit.json`, and `summary.json`
  record the evaluation boundary and source artifacts.

An independent package audit reproduced all seven confusion-matrix summaries,
all 1,464 aggregate cells (maximum absolute difference `1.42e-14`), pool and
manifest counts, array shapes and ranges, exact-match counts, and copy-risk
count/fraction arithmetic. Both PNG figures decode correctly at 1280×720.

## Claim boundary

These results support statements about frozen-CRNN compatibility,
class-conditional feature similarity, diversity, coverage, copy-risk screening,
and generation-seed stability. They do not establish waveform quality,
human-perceived realism, native generator loss, training stability, or causal
augmentation improvement. No composite score or named overall winner is used.
