# Low-resource CRNN synthetic-augmentation evaluation

This report evaluates a simulated classifier-label-scarcity setting: the CRNN
receives 50 labeled real spectrograms per species, each from a distinct
recording ID, with optional VAE-v3 or diffusion augmentation. Both generator
families consume generation-pool seeds 42, 123, 777. The VAE checkpoint
was not retrained; its filtered posterior-bank contract is
`content_safe_v2_train_filtered_existing_bank_v1` with Northern Cardinal=256, Song Sparrow=247, American Robin=256, and its three pools
were regenerated. The three diffusion pools were reused and were not regenerated. These refresh
actions come from `reports/generator_checkpoint_evaluation_2026-08-12/protocol.json` and were cross-checked
against the strict input-audit identities. The low-resource workflow itself
generated no pools.

All conditions use the same from-scratch CRNN architecture, optimizer-step
budget, 510-row generator-safe real validation set,
489-row held-out real test set, and identical post-cache
masking policy for real and generated training rows. 3 real-subset seeds
crossed with 3 classifier seeds create
9 matched blocks.
Generator-pool seeds rotate through those blocks with a Latin square. Ratios
are selected only from validation; the real-only baseline and one selected
ratio per generator reach test.

## Held-out result

| Condition | Mean test macro F1 | Paired delta vs real-only | Descriptive block-bootstrap 95% interval |
|---|---:|---:|---:|
| Real-only, 50/species | 84.75% | reference | reference |
| vae_v3 + 200/species | 87.48% | +2.72% | [+1.00%, +4.65%] |
| diffusion + 200/species | 86.21% | +1.46% | [+0.15%, +2.89%] |

- **VAE-v3 +200/species:** mean paired macro-F1 delta +2.72%; positive in 7/9 matched blocks; the descriptive interval lies entirely above zero.
- **Diffusion +200/species:** mean paired macro-F1 delta +1.46%; positive in 6/9 matched blocks; the descriptive interval lies entirely above zero.

Both validation-selected ratios are the largest tested, so the experiment identifies the best available ratio rather than a saturation point or optimum.

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
