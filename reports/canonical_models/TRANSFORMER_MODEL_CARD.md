# Rectangular Transformer model card

## Identity

- Checkpoint: `transformer_8x16_seed42.pt`
- Architecture: conditional autoregressive continuous patch Transformer
- Patch grid: 8×16 patches over the 80×256 mel array (160 patches)
- Selection: validation NLL across seeds 42, 123, and 777
- Selected seed/epoch: 42 / 51
- Validation NLL: `-1.1461628437`
- Matrix mean/median NLL: `-1.1446534` / `-1.1439769`

## Input/output contract

The model emits normalized `[1, 80, 256]` log-mel tensors. Denormalize with
the bundled training-only scaler, then decode with
`nvidia/bigvgan_v2_22khz_80band_256x` to a 22,050 Hz, 65,536-sample waveform.
The checkpoint embeds rectangular patch metadata, scaler, contract, seed,
epoch, metrics, and class order.

## Temperature sweep

Balanced 24-sample sets were decoded at temperatures 0.4, 0.6, 0.8, and 1.0.
Temperature 0.8 is the listening choice: 0.4 was visibly under-detailed,
while 1.0 increased detail and diversity but approached the upper range more
aggressively. All four sets met the finite, non-silent, and <0.1% clipping
gates; temperature was not used to select the checkpoint.

## Limitations

Validation NLL and decoder gates do not establish perceptual realism. Frozen
legacy classifier agreement is a conditioning diagnostic only. Listen to the
curated WAVs before making perceptual claims.
