# Low-resource CRNN synthetic-augmentation evaluation

This report evaluates a simulated classifier-label-scarcity setting: the CRNN
receives 50 labeled real spectrograms per species, each from a distinct
recording ID, with optional VAE-v3 or diffusion augmentation. Both generator
families reuse the existing generation-pool seeds 42, 123, 777; no generator was retrained
and no additional spectrogram was generated for this experiment.

All conditions use the same from-scratch CRNN architecture, optimizer-step
budget, real-only validation set, and identical post-cache masking policy for
real and generated training rows. 3 real-subset seeds
crossed with 3 classifier seeds create
9 matched blocks.
Generator-pool seeds rotate through those blocks with a Latin square. Ratios
are selected only from validation; the real-only baseline and one selected
ratio per generator reach test.

## Held-out result

| Condition | Mean test macro F1 | Paired delta vs real-only | Descriptive block-bootstrap 95% interval |
|---|---:|---:|---:|
| Real-only, 50/species | 84.75% | reference | reference |
| vae_v3 + 200/species | 87.69% | +2.93% | [+1.43%, +4.54%] |
| diffusion + 200/species | 85.17% | +0.42% | [-1.19%, +1.85%] |

Within this design, VAE-v3 +200/species
supports a repeatable classifier-utility improvement: its macro-F1 delta was
positive in 8/9 matched
blocks. Diffusion +200/species does not
show the same stability: its delta was positive in
5/9 blocks and its
descriptive interval spans zero. Both selected ratios are the largest tested,
so the experiment identifies the best available ratio, not a saturation point
or optimum.

The interval is a deterministic bootstrap over the 9 matched
experiment blocks, not a test-recording confidence interval or a p-value.
Every per-block result remains in `test_per_run.csv` and `paired_deltas.csv`;
seed disagreement is not hidden by the mean. The real-data subsets overlap
because the source split has only 56 Robin, 63 Cardinal, and 95 Sparrow
recording IDs; `subset_overlap.csv` quantifies that dependence.

## Interpretation boundary

This experiment can support a claim about classifier label scarcity with access
to already-trained generators. It does not establish performance for genuinely
rare or unseen species: the evaluated classes are common in this project, the
generators were trained with more source data than the 50
classifier-visible examples, and the project test set has prior evaluation
history. It also does not establish waveform quality or human-perceived realism.

## Package contents

- `protocol.json`: predeclared design, seed routing, selection, and test policy.
- `input_audit.json`: split isolation, subset counts, pool identities, and all
  3,600 generated-array audit results.
- `provenance.json`: exact input, pool, source-file, and evaluation identities.
- `validation_per_run.csv` and `validation_selection.csv`: all validation evidence.
- `test_per_run.csv`, `test_aggregate.csv`, and `paired_deltas.csv`: held-out evidence.
- `paired_summary.csv`: means, sample standard deviations, ranges, and intervals.
- `subset_membership.csv` and `subset_overlap.csv`: exact real-data subsets and
  their pairwise recording-ID overlap.
- `confusion_matrices/` and `figures/`: bounded diagnostic evidence.
