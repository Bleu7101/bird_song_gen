# Classifier artifacts

Classifier evidence now has four distinct roles. Three versioned checkpoint
packages are siblings in this directory; the first downstream augmentation
study is a report. A validation sweep, a selected final checkpoint, a legacy
held-out baseline, and an augmentation comparison must not be presented as the
same experiment.

| Package | Purpose | Selection/evaluation boundary |
|---|---|---|
| [`selected_crnn`](selected_crnn/README.md) | Current selected classifier and one held-out evaluation | CRNN seed 777 selected from validation evidence, then evaluated once on 489 test clips |
| [`architecture_comparison`](architecture_comparison/README.md) | Complete four-architecture, three-seed sweep | Validation only; all 12 checkpoints are retained |
| [`Harvey_classifier`](Harvey_classifier/README.md) | Legacy residual CNN checkpoint | Previously evaluated held-out baseline used by earlier generator reports |
| [`augmentation report`](../../../reports/crnn_synthetic_augmentation_2026-08-09/README.md) | First VAE-v3/diffusion synthetic-data sweep | Ratios selected on validation; selected arms tested once and compared descriptively with the unmatched historical selected CRNN |

`Harvey_classifier` in this table is a retained artifact-directory name. It
does not refer to an active branch; current repository work is on `main`.

The selected checkpoint is copied unchanged from
`architecture_comparison/crnn/seed_777/best.pt`. Earlier generated-sample
reports continue to name the legacy residual checkpoint explicitly so their
recorded metrics remain reproducible.

Do not compare all architecture checkpoints on the held-out test split. The
test split is for the predeclared selected candidate only.

An exact-content audit performed after these historical runs found nine of the
519 validation clips duplicated byte-for-byte in training; no exact duplicate
reaches the 489-clip test split. The recorded v1 results are retained as-is.
Future experiments should use `manifests/content_safe_v2/` and the
manifest-backed cache.
