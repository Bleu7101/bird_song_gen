# VAEv3 Notebook Evaluation Pool Design

## Goal

Recreate the missing 48-sample VAEv3 generated pool from the existing
`conditional_vae_v3_best.pt` using the sampling logic in
`notebooks/04_conditional_vae.ipynb`, without modifying the original notebook,
the historical output directory, or Harvey's generator code.

## Fixed protocol

- Source notebook: the local `04_conditional_vae.ipynb`.
- Checkpoint: the existing VAEv3 best checkpoint.
- Posterior bank: the existing bank fitted on `train` only; it is not refit by
  this run.
- Class order: Northern Cardinal, Song Sparrow, American Robin.
- Samples: 16 per species, 48 total, matching the missing historical manifest.
- Generation seed: notebook seed `42`, with the notebook's generation stream
  seed `42 + 100`.
- Posterior temperature: `0.35`.
- Generated source domain: global standardized log-mel.
- Derived classifier-input arrays: saved only as an auxiliary artifact using
  the notebook's old relative-dB adapter.
- Test data: not loaded by the pool-generation notebook execution; it is read
  only by the separate frozen MSE evaluation.

## Execution boundary

The runner reads the source notebook, executes only the cells needed to load
the model and run the generation function, and injects path/config overrides.
It skips training, test reconstruction metrics, test plots, latent inspection,
and report-summary cells. The executed notebook is saved with the pool for
auditability.

All outputs are written to a new, non-empty-output-refusing run directory.
The pool manifest uses paths relative to that directory. Metadata records the
source notebook hash, checkpoint and posterior-bank hashes, Git revision,
sampling constants, class map, array contract, and hashes for every generated
array.

## Validation

Unit tests cover non-overwriting, portable manifest conversion, metadata
contract, and notebook-cell selection/config injection. The run validates 48
standardized arrays and 48 auxiliary classifier-input arrays before invoking
the Generated-to-test MSE evaluator in standardized-logmel space.
