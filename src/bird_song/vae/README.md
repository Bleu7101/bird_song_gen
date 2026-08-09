# VAE experiments on main

VAE v1, v2, and v3 are preserved on `main`. The experiment entry point is
`notebooks/04_conditional_vae.ipynb`; versioned checkpoints are under
`artifacts/vae_artifacts/conditional_vae*/`, and recorded outputs are under
`outputs/conditional_vae*/`.

This package directory remains a namespace placeholder rather than a second
VAE implementation. Use the notebook and its recorded artifacts together so
the architecture version, preprocessing, checkpoint path, and evaluation
evidence stay aligned. A separate VAE branch is no longer required because all
three recorded versions are retained on `main`.
