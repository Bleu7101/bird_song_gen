# Classifier artifacts

This directory separates three different evidence roles. The folders are
siblings deliberately: a validation sweep, a selected final checkpoint, and a
legacy held-out baseline must not be presented as the same experiment.

| Package | Purpose | Selection/evaluation boundary |
|---|---|---|
| [`selected_crnn`](selected_crnn/README.md) | Current selected classifier and one held-out evaluation | CRNN seed 777 selected from validation evidence, then evaluated once on 489 test clips |
| [`architecture_comparison`](architecture_comparison/README.md) | Complete four-architecture, three-seed sweep | Validation only; all 12 checkpoints are retained |
| [`Harvey_classifier`](Harvey_classifier/README.md) | Legacy residual CNN checkpoint | Previously evaluated held-out baseline used by earlier generator reports |

The selected checkpoint is copied unchanged from
`architecture_comparison/crnn/seed_777/best.pt`. Both copies must retain the
same SHA-256 hash. Earlier generated-sample reports continue to name the
legacy residual checkpoint explicitly so their recorded metrics remain
reproducible.

Do not compare all architecture checkpoints on the held-out test split. The
test split is for the predeclared selected candidate only.
