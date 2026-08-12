# First CRNN synthetic-augmentation evaluation

## Technical summary

This first evaluation did **not** show a mean held-out improvement over the
historical selected CRNN. Validation selected 200 generated samples per species
for both VAE v3 and diffusion. Across CRNN seeds 42, 123, and 777, the selected
VAE-v3 arm reached 89.48% mean test macro F1 and the selected diffusion arm
reached 89.28%. Those values are 0.68 and 0.88 percentage points below the
historical seed-777 CRNN result of 90.16%.

This is a descriptive comparison, not a causal estimate of synthetic
augmentation. The reference is one historical seed trained through the earlier
on-the-fly WAV path, whereas each augmented arm used three newly trained seeds
with fixed cached real spectrograms plus generated arrays. No matched
three-seed cached-real-only arm was recorded in this sweep. The supported
verdict is therefore:

> Neither generated-data arm beat the historical selected CRNN on mean test
> macro F1 in this first evaluation. The experiment does not establish that
> generated data harms performance.

## Selected test results

| Condition | Test runs | Mean accuracy | Mean macro F1 | Accuracy delta | Macro-F1 delta |
|---|---:|---:|---:|---:|---:|
| Historical selected CRNN, seed 777 | 1 | 89.98% | 90.16% | reference | reference |
| VAE v3, 200 generated/species | 3 | 89.43% | 89.48% | -0.55 pp | -0.68 pp |
| Diffusion, 200 generated/species | 3 | 89.23% | 89.28% | -0.75 pp | -0.88 pp |

Seed-to-seed variation was material:

| Generator | Seed | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| VAE v3 | 42 | 92.43% | 92.50% |
| VAE v3 | 123 | 87.12% | 87.08% |
| VAE v3 | 777 | 88.75% | 88.87% |
| Diffusion | 42 | 89.98% | 90.23% |
| Diffusion | 123 | 87.73% | 87.54% |
| Diffusion | 777 | 89.98% | 90.09% |

The individual seed results make the three-seed mean essential. A favorable
single seed would overstate the evidence. Across the three test runs, the
sample standard deviations were 2.72 percentage points for VAE-v3 accuracy and
2.76 points for macro F1; the diffusion standard deviations were 1.30 and 1.51
points, respectively.

## Validation selected the largest tested pool

Each ratio is the number of generated arrays added per species. The selected
ratio maximized mean best-checkpoint validation macro F1 across the three CRNN
seeds.

| Generator | 50/species | 100/species | 200/species | Selected |
|---|---:|---:|---:|---:|
| VAE v3 | 88.73% | 88.15% | 89.23% | 200 |
| Diffusion | 87.25% | 89.00% | 89.32% | 200 |

Only the selected ratio was evaluated on the 489-clip test manifest. Exact
per-run validation and test values are in
[validation_summary.csv](validation_summary.csv) and
[test_summary.csv](test_summary.csv).

The validation advantage of the selected ratio over the runner-up was only
0.50 percentage points for VAE v3 and 0.32 points for diffusion, smaller than
the variation across CRNN seeds. Selection of 200/species is the recorded v1
choice, not evidence of a stable optimum beyond the tested ratios and pool.

## Scope and metric definitions

- The classifier predicts American Robin, Northern Cardinal, or Song Sparrow.
- The historical v1 manifests are recording-ID-isolated and contain 2,339
  training clips, 519 validation clips, and 489 test clips.
- Accuracy is correct predictions divided by clips.
- Macro F1 is the unweighted mean of the three species-level F1 values.
- A ratio of 200 means 600 generated arrays total, 200 for each species. The
  selected training sets therefore contained 2,339 real and 600 generated
  rows, for 2,939 rows and a 20.42% synthetic row share.
- The reported augmented means are arithmetic means across CRNN seeds 42, 123,
  and 777. The historical reference is a single preselected seed-777
  checkpoint, not the mean of a matched control arm.

## Experimental design

Every augmented run used the 404,451-parameter CRNN with width 32 and dropout
0.30, batch size 64, 1,440 optimizer steps, and validation every 36 steps. The
best checkpoint within each run was selected by validation macro F1. Ratios
50, 100, and 200 were compared by their three-seed mean validation macro F1;
only the selected ratio proceeded to test evaluation. All 18 retained run
configs record `cuda`, confirming that the sweep trained on GPU.

Real inputs came from the historical v1 logical rows in the cache at
`artifacts/spectrograms/`. The recorded cache audit found 3,347 logical rows
referencing 3,323 physical arrays. Generated inputs came from the reusable local pools
at:

- `artifacts/generated_spectrograms/vae_v3/manifest.csv`
- `artifacts/generated_spectrograms/diffusion/manifest.csv`

Both manifests contain 600 classifier-ready float32 arrays with shape
`[1,128,128]`, range `[-1,1]`, and 200 arrays per species. At packaging time,
all 1,200 arrays existed and passed the recorded shape, dtype, and range checks.

The VAE pool used the committed V3 checkpoint and posterior bank at temperature
0.35. The diffusion pool used the recorded EMA checkpoint with deterministic
DDIM sampling: 100 reverse steps, eta 0, guidance weight 3.0, and clean-sample
clamping at 4 standardized units. Labels enter classifier training by species
name, so the generators' numeric class order is not reused as the CRNN order.

See [protocol.json](protocol.json) for the complete machine-readable design and
[provenance.json](provenance.json) for source identities and input checks.

## Limitations and interpretation boundary

1. **The baseline is unmatched.** The historical reference is one selected
   seed, while each augmented result is a three-seed mean. A matched
   cached-real-only run for seeds 42, 123, and 777 is required to estimate the
   augmentation effect.
2. **The training input paths differ.** The augmented arms trained on fixed
   cached real spectrograms. The historical checkpoint was trained through the
   earlier on-the-fly WAV path with stochastic training transforms. The
   comparison therefore combines synthetic-data and input-pipeline changes.
3. **The same test set has prior project history.** These 489 clips were used
   for the previously recorded selected-CRNN evaluation. The new ratios were
   selected on validation, but this test result should now be treated as
   descriptive evidence and not used for further tuning.
4. **Generator-pool uncertainty is not measured.** Classifier seeds vary, but
   both experiments reuse one seed-42 generated pool. A different generated
   pool could change the result.
5. **The historical validation split is not exact-content-isolated.** A
   post-run audit found nine cross-split content groups involving 26 logical
   rows: nine of 519 validation clips also occur byte-for-byte in training. No
   exact duplicate reaches test, so the held-out metrics above are not directly
   leaked, but validation and ratio selection can be optimistic.
6. **The retained run metadata is incomplete.** The local run configs do not
   record an optimizer specification or a committed run-code revision. This
   package preserves the recorded outputs and metadata without inventing those
   missing fields.

## Future experiment boundary

Future experiments should use `manifests/content_safe_v2/`. Those versioned
manifests preserve all 519 validation and 489 test clips while retaining 2,315
unique-content training rows, and the current cache can serve those rows without
duplicating physical arrays. A future augmentation-only sweep should also vary
the generated-pool seed and select ratios without reusing these test outcomes.

A causal claim about augmentation would still require a matched cached-real-only
control with the same seeds and step budget. No such arm was requested or
recorded for v1, and this package does not schedule one. Until one exists, this
is a useful first evaluation showing no demonstrated mean gain, not evidence
for or against a causal augmentation effect.

## Package contents

| File | Role |
|---|---|
| [summary.json](summary.json) | Compact result and verdict |
| [protocol.json](protocol.json) | Dataset, representation, pools, and selection design |
| [validation_summary.csv](validation_summary.csv) | Exact per-seed validation evidence for all ratios |
| [test_summary.csv](test_summary.csv) | Historical reference, exact per-seed tests, and augmented means |
| [provenance.json](provenance.json) | Source identities, pool checks, and known provenance gaps |
| [cache_audit.json](cache_audit.json) | Physical deduplication and exact-content split audit |
