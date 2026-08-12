# Three-seed checkpoint generator evaluation

This package is the bounded evidence record for inference-only evaluation of
the frozen VAE-v3 and conditional-diffusion checkpoints. The existing seed-42
pools are preserved; seed-123 and seed-777 pools are generated locally under
ignored `runs/generator_checkpoint_evaluation/pools/` and are not copied into
the repository.

The primary evaluator is the selected CRNN (`selected_crnn/best.pt`). The
legacy residual CNN is a sensitivity check because earlier generator reports
used it. Both evaluators use their embedded classifier configurations on the
489-clip `content_safe_v2` test set. Feature standardization and 64-component
PCA are fit separately per evaluator using only the 2,315 content-safe training
clips. Copy-risk thresholds are established from validation-to-training
nearest-neighbor distances before generated samples are inspected.

Each model has three deterministic 200-per-species pools (1,800 arrays/model).
The diffusion sampler uses the checkpoint EMA weights, the cosine 1,000-step
schedule, DDIM with 100 steps and `eta=0`, guidance 3.0, and predicted-`x0`
clamp 4.0. VAE-v3 uses the recorded 256-anchor posterior bank at temperature
0.35. Standardized outputs are converted with the recorded mean/std, per-sample
maximum subtraction, `[-80, 0]` clipping, and classifier `[-1, 1]` scaling.

`metrics_per_seed.csv` retains every model/seed/classifier/species result;
`metrics_aggregate.csv` reports mean, sample standard deviation, and range over
the three seeds. `classifier_scores.csv` and `confusion_matrices/` contain
conditioning diagnostics. `nearest_neighbor_summary.csv` records copy-risk and
generated-to-training distances. Feature FID-style distances are named
`frechet_*` because they are distances in frozen classifier feature space, not
audio FAD. Resampled metrics use 200 deterministic resamples of 128 real and
128 generated examples per species.

The diffusion seed-42 pool has much lower target-label agreement than its
seed-123 and seed-777 counterparts in this classifier view. That disagreement is
retained explicitly as generation-seed instability; it is not hidden by pooling
the arrays and is not converted into a model ranking.

These results support classifier compatibility, conditioning, classifier-feature
distribution similarity, diversity, coverage, copying risk, and seed stability.
They do not establish waveform quality, human-perceived realism, native
generator loss, training stability, or causal augmentation improvement. If the
two evaluators rank models differently, that is reported as encoder dependence;
there is no composite score or named winner.

The notebook `notebooks/generator_checkpoint_evaluation.ipynb` reads only this
package and runs without checkpoints or ignored pools. `protocol.json` and
`provenance.json` preserve the inference boundary; the external diffusion
checkpoint is neither copied nor tracked.
