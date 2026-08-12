# Prompt for drafting the Generated-to-test MSE section

Copy the prompt below into the report-outline drafting conversation. It tells the drafting model where the verified data live, which files have priority, and which claims are out of scope.

```text
You are drafting the Generated-to-test MSE subsection of the bird-song project report outline. Work from the repository root and use only the verified local artifacts listed below. Do not invent values, replace missing values with estimates, or silently use a different generated pool.

## Read these files first

1. `doc/Project_Report_Template/Generated_to_test_MSE_reference.md`
2. `reports/generated_to_test_mse_v1/vae_v3_notebook_seed42/summary.csv`
3. `reports/generated_to_test_mse_v1/vae_v3_notebook_seed42/protocol.json`
4. `runs/generated_to_test_mse_v1/vae_v3_notebook_seed42/generation.json`
5. `reports/generated_to_test_mse_v1/vae_v3_notebook_seed42/per_sample.csv`
6. `runs/generated_to_test_mse_v1/vae_v3_notebook_seed42/manifest.csv`
7. `runs/generated_to_test_mse_v1/vae_v3_notebook_seed42/array_hashes.csv`
8. `doc/Project_Report_Template/Report_Outline.md`

Use the implementation only to understand definitions if needed:
`src/bird_song/evaluation/generated_to_test_mse.py`,
`scripts/evaluate_generated_to_test_mse.py`, and
`docs/superpowers/specs/2026-08-12-vae-v3-notebook-pool-design.md`.

## Source priority

- Use `summary.csv` for all reported numerical values.
- Use `protocol.json` for the MSE definitions, numerical-domain contract, input hashes, and test-set policy.
- Use `generation.json` for the VAEv3 checkpoint, posterior bank, notebook, seed, temperature, device, and pool provenance.
- Use `per_sample.csv` only for individual examples, nearest-neighbor paths, or an appendix-level distribution audit.
- Treat any older wording in `Report_Outline.md` saying that generated-to-test MSE is “not yet computed” as stale after the files above have been verified.

## Facts that must appear in the draft

- The comparison is between the existing VAEv3 generated spectrogram pool and the local held-out test set.
- The generated pool is `runs/generated_to_test_mse_v1/vae_v3_notebook_seed42/`, made from `notebooks/04_conditional_vae.ipynb` and the existing `conditional_vae_v3_best.pt` checkpoint.
- The pool uses seed 42, generation stream seed 142, temperature 0.35, 16 samples per species, 48 generated samples total, CUDA, and a posterior bank fitted on the train split only.
- The test manifest is `artifacts/spectrograms/manifests/spectrogram_test.csv`, with 489 samples: American Robin 145, Northern Cardinal 162, and Song Sparrow 182.
- If that logical test-manifest path is absent in the current worktree, follow the exact local path recorded in `protocol.json` under `inputs.test_manifest`; do not substitute another split. Keep the absolute machine path out of report prose.
- Both generated and test arrays were compared in `standardized_logmel_float32` with shape `[1,128,128]`; no conversion was needed for the primary MSE.
- Auxiliary `classifier_input` arrays exist in the pool but were not used for this MSE. Do not describe the comparison as being in the current classifier-input domain.
- The test set was used only for the final frozen evaluation, not for posterior-bank fitting, sample selection, temperature tuning, or checkpoint selection.

## Metric wording

Define generated-to-test MSE as follows: for each generated spectrogram, find the pixel-MSE-nearest test spectrogram of the same species, then summarize those 48 nearest-neighbor distances.

Define the real-to-real baseline as follows: for each test spectrogram, find the nearest *other* test spectrogram of the same species, then summarize those 489 leave-one-out distances.

State that the reported summaries include `count`, `mean`, `std`, `median`, `Q1`, `Q3`, `min`, and `max`, both overall and per species. The generated and real baseline counts differ, so this is not a paired comparison.

## Numerical values to report

Read the exact values from `summary.csv`. In normal report prose, round to four decimal places:

- Generated-to-test overall: count 48, mean 0.5120, std 0.1998, median 0.4501, Q1 0.3878, Q3 0.6231, min 0.2054, max 1.2559.
- Generated-to-test American Robin: count 16, mean 0.4308, std 0.1122, median 0.4153, Q1 0.3692, Q3 0.4632, min 0.2637, max 0.6913.
- Generated-to-test Northern Cardinal: count 16, mean 0.5579, std 0.1689, median 0.5849, Q1 0.4442, Q3 0.6602, min 0.2054, max 0.8250.
- Generated-to-test Song Sparrow: count 16, mean 0.5474, std 0.2708, median 0.4488, Q1 0.3865, Q3 0.6395, min 0.2438, max 1.2559.
- Real-to-real overall: count 489, mean 0.3350, std 0.1419, median 0.3185, Q1 0.2368, Q3 0.4154, min 0.0063, max 0.9368.
- Real-to-real American Robin: count 145, mean 0.3237, std 0.1310, median 0.3019, Q1 0.2343, Q3 0.4256, min 0.0969, max 0.7061.
- Real-to-real Northern Cardinal: count 162, mean 0.3650, std 0.1816, median 0.3435, Q1 0.2290, Q3 0.4625, min 0.0063, max 0.9368.
- Real-to-real Song Sparrow: count 182, mean 0.3175, std 0.1005, median 0.3064, Q1 0.2421, Q3 0.3756, min 0.1419, max 0.7630.

## Interpretation limits

Use this interpretation: the generated-to-test mean is higher than the real-to-real reference in this pixel-space representation, but the result is only a same-species unpaired nearest-neighbor proximity diagnostic. It does not establish perceptual realism, audio quality, or downstream usefulness. A very low nearest-neighbor MSE can reward copying or limited diversity, so do not present MSE alone as a realism score.

Do not claim that:

- Harvey's corrected canonical pool was used;
- Harvey's current CRNN teacher/classifier was used for the comparison;
- the arrays were converted to the current classifier input for this primary metric;
- the metric is a paired generated-versus-test reconstruction error;
- the metric proves perceptual or waveform realism;
- the test set influenced generation, temperature, posterior-bank fitting, sample selection, or checkpoint choice.

## Required output

Return a concise English report-outline insertion for Section 6.2 with:

1. one paragraph defining the protocol and domain;
2. one compact table or two small tables containing overall and per-species results;
3. one provenance paragraph naming the pool, checkpoint/notebook, seed, temperature, and train-only posterior bank;
4. one limitation paragraph containing the pixel-space and copying caveats;
5. a short list of which old “not yet computed” checklist/open-item entries should be marked complete.

Use repository-relative Markdown links for evidence. Do not edit code or result files while drafting. If any required source file is missing, stop and report the exact missing path instead of fabricating a result.
```

## Expected handoff

The drafting model should produce report prose and an outline update, not a new evaluation. The evaluation artifacts remain the source of truth; this prompt is only a navigation and claim-boundary aid.
