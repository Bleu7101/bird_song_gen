# Generator-only speed comparison

This matched benchmark compares fresh in-memory VAE-v3 and diffusion generation
for 600 spectrograms per repeat
(200 per species). Both models use FP32, batch size
8, the same CUDA device, 5
warm-up batches, and 5 synchronized repeats.

| Model | Mean seconds / 600 | Sample SD | Median [Q1, Q3] seconds | Mean spectrograms/s | Peak CUDA memory |
|---|---:|---:|---:|---:|---:|
| VAE-v3 | 0.3058 | 0.0096 | 0.3046 [0.2983, 0.3076] | 1963.276 | 0.185 GiB |
| Diffusion | 412.3090 | 2.8120 | 411.1814 [410.9396, 411.2319] | 1.455 | 0.333 GiB |

Diffusion took **412.0032 seconds
longer** per equal-count request and **1348.09x**
the VAE-v3 generator time in this environment.

## Boundary

This is generator-only spectrogram latency. It includes posterior-anchor and
latent sampling plus VAE decoding, or initial-noise construction plus 100-step
DDIM sampling. It excludes checkpoint and posterior-bank loading, warm-up,
classifier-scale conversion, CPU transfer, array serialization, report I/O,
waveform decoding, and WAV writing. Runtime is reported separately from quality
and downstream CRNN utility.

The models were benchmarked sequentially rather than interleaved. Each repeat
reuses the same deterministic sample streams, so its variation measures system
runtime variation rather than variation across generated sample populations.

Measurement sequence note: The completed diffusion benchmark was retained; VAE-v3 was remeasured afterward using the filtered 759-anchor bank. No diffusion sampling was rerun.


Raw repeat-level measurements are in `repeat_results.csv`; the complete captured
environments and model settings are in `vae_v3_benchmark.json` and
`diffusion_benchmark.json`.
