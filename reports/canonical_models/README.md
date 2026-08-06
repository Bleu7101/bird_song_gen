# Canonical BigVGAN generator bundle

This is the small publication bundle for `decoder_test`. It contains only the
validation-selected WGAN-GP and Transformer generator checkpoints, their
configs/contracts, and twelve curated WAVs (two per species per generator).

The checkpoints are CPU-loadable PyTorch files and embed the model config,
seed, selected epoch, validation metrics, class order, training-only scaler,
and BigVGAN contract. The full all-seed matrix and temperature sweep are in
[`../bigvgan_matrix_report.md`](../bigvgan_matrix_report.md).

## Selected models

| Model | Selection rule | Checkpoint | SHA-256 |
|---|---|---|---|
| WGAN-GP stability | Lowest validation selection error within the lower-error stability configuration | `wgan_gp_stability_seed42.pt` | `c760ad2d396f7a307d4f7db66267e22cd3e50e92c9fd608b39c3b4d6dbf67811` |
| Transformer 8×16 | Lowest validation NLL within the lower-NLL rectangular patch configuration | `transformer_8x16_seed42.pt` | `e9ac12cb8b9f4a878af4f5925789da24bbd6649a9a6f181114968601924bb84e` |

## Contract and configs

- BigVGAN: `nvidia/bigvgan_v2_22khz_80band_256x`
- Audio: 22,050 Hz, exactly 65,536 samples
- Mel arrays: raw natural-log `[80, 256]`; model tensors `[batch, 1, 80, 256]`
- Scaler: global min-max fitted on the training split only, minimum
  `-11.512925148010254`, maximum `2.1087865829467773`
- Class order: American Robin, Northern Cardinal, Song Sparrow
- WGAN config: [`wgan_stability.json`](wgan_stability.json)
- Transformer config: [`transformer_8x16.json`](transformer_8x16.json)
- Vocoder contract: [`vocoder_contract.json`](vocoder_contract.json)
- Training scaler: [`scaler.json`](scaler.json)

## Curated listening set

The WAVs are balanced and use the same frozen BigVGAN decoder. See
[`listening_manifest.csv`](listening_manifest.csv) for model, species, sample,
sample rate, length, and SHA-256. Classifier outputs are not included in the
selection rule and do not establish perceptual realism.

BigVGAN weights, resumable checkpoints, mel caches, bulk arrays, and bulk
audio remain excluded from Git.
