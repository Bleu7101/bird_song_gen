# Conditional Diffusion for Bird-Song Generation

## Branch scope

This README documents the `diffusion_vincent` branch. The branch is an independent diffusion-model development track separated from `main`; its notebook configuration, checkpoints, generated samples, and evaluation outputs should not be assumed to match the current state of `main`.

The goal of this branch is to reproduce the class-conditional diffusion pipeline for three bird species:

- American Robin
- Northern Cardinal
- Song Sparrow

The primary implementation is [`notebooks/05_conditional_diffusion.ipynb`](notebooks/05_conditional_diffusion.ipynb). This README intentionally does not document the implementations in Notebooks 1--4. Notebooks 1--3 must be run as prerequisites, while Notebook 4 is the separate VAE branch and is not required to reproduce the diffusion results.

## Repository layout

```text
notebooks/
|-- 01_dataset_audit.ipynb                  Required prerequisite
|-- 02_preprocess_logmel.ipynb              Required prerequisite
|-- 03_classifier.ipynb                     Required prerequisite
`-- 05_conditional_diffusion.ipynb          Diffusion training, sampling, and export

checkpoints/conditional_diffusion/
`-- conditional_diffusion_best_low_batch_100_eps_weighted.pt
                                               Selected diffusion checkpoint

classifier_artifacts/Harvey_classifier/
`-- best.pt                                 Frozen Residual CNN checkpoint

scripts/
|-- export_classifier_scale_and_audio.py    Export classifier-scale NPY and WAV files
|-- rescale_generated_and_classify.py       Scale-conversion/classifier sanity check
`-- 06_evaluate_generated.py                Formal generated-sample classifier report

outputs/conditional_diffusion/
|-- generated_npy/                          Standardized log-mel samples from Notebook 5
|-- generated_npy_classifier_scale/         Converted NPY files for the classifier
|-- generated_audio/                        Griffin--Lim WAV reconstructions
|-- conditional_samples.png                 Generated-spectrogram panel
|-- denoising_trajectory.png                Predicted x0 reverse trajectory
|-- generated_manifest.csv                  Generated-sample manifest
|-- run_summary.json                        Diffusion run configuration summary
`-- *.txt                                   Saved classifier/development reports
```

Generated data, processed tensors, and evaluation runs are ignored by Git in some environments. Their directories may therefore be created only after the relevant notebooks or scripts have been run.

## Environment setup

Python 3.10 or newer is required. A CUDA GPU is strongly recommended for diffusion training and sampling.

The selected diffusion checkpoint is stored with Git LFS:

```bash
git lfs install
git lfs pull
```

Install the Python dependencies and the local `bird_song` package from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Reproduction order

### 1. Run the prerequisite notebooks

Run the following notebooks in order using **Run All**:

1. `notebooks/01_dataset_audit.ipynb`
2. `notebooks/02_preprocess_logmel.ipynb`
3. `notebooks/03_classifier.ipynb`

These notebooks prepare the manifests, standardized `1 x 128 x 128` log-mel tensors, normalization statistics, and classifier checkpoint required by the diffusion and evaluation code. Their internal implementation is outside the scope of this branch README.

Do **not** run `notebooks/04_conditional_vae.ipynb` for diffusion reproduction. The VAE is a separate generator and is not an input to Notebook 5.

### 2. Run the conditional diffusion notebook

Open and run:

```text
notebooks/05_conditional_diffusion.ipynb
```

Notebook 5 defines a class-conditional U-Net diffusion model with:

- a 1,000-step cosine diffusion schedule;
- timestep and species conditioning through FiLM residual blocks;
- self-attention at the `16 x 16` resolution;
- classifier-free guidance;
- EMA weights for sampling;
- 100-step deterministic DDIM sampling by default.

The selected checkpoint is:

```text
checkpoints/conditional_diffusion/conditional_diffusion_best_low_batch_100_eps_weighted.pt
```

To reproduce samples from this checkpoint rather than retrain the model, verify the following settings in the notebook configuration cell:

```python
RUN_TRAINING = False
SAMPLER = "ddim"
DDIM_STEPS = 100
GUIDANCE_WEIGHT = 3
SAMPLES_PER_SPECIES = 8
```

The full checkpoint was trained with `BASE_CHANNELS=64` and `NUM_TIMESTEPS=1000`. If the notebook is run without CUDA, set `CPU_AUTO_SCALE=False` before model construction; otherwise the automatic reduced CPU architecture will not match the supplied checkpoint. Full-size diffusion sampling on CPU may be slow.

Before continuing, confirm that Notebook 5 created:

```text
outputs/conditional_diffusion/generated_npy/
outputs/conditional_diffusion/generated_manifest.csv
outputs/conditional_diffusion/conditional_samples.png
outputs/conditional_diffusion/denoising_trajectory.png
```

The files in `generated_npy/` are standardized generator-domain log-mel tensors. They must not be passed directly to the classifier, which expects a different `[-1,1]` normalization convention.

## Convert generated NPY files

Run the export script from the repository root:

```bash
python scripts/export_classifier_scale_and_audio.py
```

For every standardized generated NPY file, the script creates two outputs from the same sample:

```text
outputs/conditional_diffusion/generated_npy_classifier_scale/*.npy
outputs/conditional_diffusion/generated_audio/*.wav
```

The first output converts the generated spectrogram to the classifier's `[-1,1]` scale. The second converts it back to relative decibels and mel power before Griffin--Lim waveform reconstruction. Audio is not reconstructed from the classifier-scale tensor.

## Evaluate generated samples

### Scale-conversion sanity check

Run:

```bash
python scripts/rescale_generated_and_classify.py
```

This diagnostic converts the generated tensors in memory, evaluates them with the frozen classifier, and applies the same conversion to a real test-set control. It prints the predicted-class counts, mean confidence, and target-label accuracy. It does not create the classifier-scale NPY files or WAV files; those are created by `export_classifier_scale_and_audio.py`.

### Formal classifier report

On Linux, macOS, or a Bash-compatible shell, run the supplied command:

```bash
PYTHONPATH=src python scripts/06_evaluate_generated.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --input outputs/conditional_diffusion/generated_npy_classifier_scale --name diffusion_run1
```

In Windows PowerShell, the equivalent commands are:

```powershell
$env:PYTHONPATH = "src"
python scripts/06_evaluate_generated.py --checkpoint classifier_artifacts/Harvey_classifier/best.pt --input outputs/conditional_diffusion/generated_npy_classifier_scale --name diffusion_run1
```

If the project was installed with `python -m pip install -e .`, the `PYTHONPATH` assignment is normally unnecessary.

The evaluation writes:

```text
runs/evaluation/generated_classifier_scores.csv
runs/evaluation/generated_classifier_scores.summary.json
outputs/conditional_diffusion/diffusion_run1.txt
```

The CSV contains per-file predictions and probabilities. The JSON contains aggregate statistics, and the text file provides a human-readable copy of the report. Existing development and classifier reports can also be found under `outputs/conditional_diffusion/`.

Because the converted NPY files are stored in one flat directory, the formal command above reports confidence and predicted-class counts but does not infer intended labels from parent folders. The sanity-check script infers intended species from filenames when reporting target-label accuracy.

## Important evaluation note

The Residual CNN and diffusion model were trained from the same underlying bird-song dataset. Classifier confidence is therefore an internal class-adherence signal, not an independent measure of realism. Generated WAV files should also be checked through listening, spectrogram inspection, or an external recognizer such as Merlin Bird ID. Griffin--Lim artifacts and the small number of generated samples should be considered when interpreting these results.

## Selected configuration

| Setting | Value |
|---|---:|
| Input | Standardized `1 x 128 x 128` log-mel tensor |
| Target species | 3 |
| Diffusion steps | 1,000 |
| Noise schedule | Cosine |
| Base channels | 64 |
| Attention resolution | `16 x 16` |
| Classifier-free label dropout | 0.1 |
| Guidance weight | 3 |
| Default sampler | DDIM |
| DDIM sampling steps | 100 |
| EMA decay | 0.999 |
| Generated samples | 8 per species |

The training data loader uses inverse-frequency weighted sampling to reduce class imbalance. This training sampler is separate from the classifier-free guidance weight used during generation.
