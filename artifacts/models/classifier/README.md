# Classifier artifacts

Classifier evidence has three distinct roles. The validation sweep, selected
final checkpoint, and legacy held-out baseline must not be presented as the
same experiment.

| Package | Purpose | Selection/evaluation boundary |
|---|---|---|
| [`selected_crnn`](selected_crnn/README.md) | Current selected classifier and one held-out evaluation | CRNN seed 777 selected from validation evidence, then evaluated once on 489 test clips |
| [`architecture_comparison`](architecture_comparison/README.md) | Complete four-architecture, three-seed sweep | Validation only; all 12 checkpoints are retained |
| [`Harvey_classifier`](Harvey_classifier/README.md) | Legacy residual CNN checkpoint | Historical real-audio held-out baseline; not used for maintained generated-sample evaluation |

`Harvey_classifier` in this table is a retained artifact-directory name. It
does not refer to an active branch; current repository work is on `main`.

The selected checkpoint is copied unchanged from
`architecture_comparison/crnn/seed_777/best.pt`. Maintained generated-sample
evaluation uses this selected CRNN only.

Do not compare all architecture checkpoints on the held-out test split. The
test split is for the predeclared selected candidate only.

An exact-content audit performed after these historical runs found nine of the
519 validation clips duplicated byte-for-byte in training; no exact duplicate
reaches the 489-clip test split. The recorded v1 results are retained as-is.
Future experiments should use `manifests/content_safe_v2/` and the
manifest-backed cache.
