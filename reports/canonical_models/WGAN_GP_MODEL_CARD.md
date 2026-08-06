# WGAN-GP model card

## Identity

- Checkpoint: `wgan_gp_stability_seed42.pt`
- Architecture: conditional WGAN-GP mel generator
- Selection: stability configuration, seed 42, epoch 18
- Validation selection error: `0.0441303463`
- Matrix: stability mean `0.0955538`, median `0.0778064`; current mean
  `0.1267609`, median `0.1182563`

## Input/output contract

The model emits normalized `[1, 80, 256]` log-mel tensors. Denormalize with
the bundled training-only scaler, then decode with
`nvidia/bigvgan_v2_22khz_80band_256x` to a 22,050 Hz, 65,536-sample waveform.
The checkpoint embeds this contract and the three-species class order.

## Recorded gates

The selected seed generated 24 balanced samples (8 per species): all mels and
waveforms were finite, there were zero silent samples, mean mel saturation was
`0.0`, and maximum waveform clipping was `0.000290` (<0.001).

## Limitations

Validation detail ratios and decoder validity establish an interface check,
not acoustic realism. Frozen classifier agreement is reported separately as a
conditioning diagnostic. Listen to the curated WAVs before making perceptual
claims.
