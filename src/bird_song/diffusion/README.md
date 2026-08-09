# Diffusion branch guide

`main` does not select or package a canonical diffusion model. Diffusion-model
implementations and experiment artifacts are maintained on the separate
`Diffusion`, `Difussion`, and `diffusion_vincent` branches.

This directory is only a namespace placeholder on `main`; it should not be
interpreted as the implementation for those branches. Use the README,
notebooks, source, configs, checkpoints, and recorded evidence from the
specific diffusion branch being reviewed. Before comparing a diffusion result
with a `main` generator, verify that preprocessing, class order, dataset split,
checkpoint format, and decoder path match.
