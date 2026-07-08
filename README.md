# Bird Song Generation

Generative modeling project for bird song audio. The current repository structure focuses on a reproducible setup, dataset audit notebook, target-species split manifests, and a small demo script. The audio dataset itself is expected to be available locally and is not committed to this repository.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- demo.py
|-- 01_dataset_audit.ipynb
|-- doc/
|   `-- Proposal_Generative Modeling of Toronto Bird Songs Using VAEs and Diffusion Models.pdf
|-- manifests/
|   |-- full_dataset_manifest.csv
|   |-- full_dataset_train.csv
|   |-- full_dataset_validation.csv
|   `-- full_dataset_test.csv
`-- bird_songs_dataset/        # local only, not committed
    |-- bird_songs_metadata.csv
    `-- wavfiles/
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Place the dataset at the project root:

```text
bird_songs_dataset/
|-- bird_songs_metadata.csv
`-- wavfiles/
```

Current target species:

- Northern Cardinal
- Song Sparrow
- American Robin

The manifest CSVs reference WAV files by relative path, for example `wavfiles/70119-0.wav`. This assumes each teammate has the same local dataset layout.

## Current Workflow

The project is organized around five planned stages:

1. Read the dataset and create train/validation/test splits.
2. Convert WAV files to spectrograms.
3. Train a supervised species classifier.
4. Train a VAE on spectrograms.
5. Train a diffusion model on spectrograms.

Only Step 1 has been prepared so far. The audit notebook `01_dataset_audit.ipynb` creates split manifests for the three target species and keeps clips from the same source recording id in the same split to reduce leakage.

## Usage

Run the audit notebook first if you need to regenerate the split manifests. Then run the demo to verify that the local dataset and split manifests are readable:

```bash
python demo.py
```

Example output after generating the three-species manifests:

```text
Metadata rows: 5422
Manifest rows: 3347
Species: 3
Splits: test, train, validation
Sample: Song Sparrow -> bird_songs_dataset/wavfiles/...
```

## Notes

- The audio dataset is not committed because it is large and should be shared separately.
- Generated spectrograms, checkpoints, model files, and run logs are ignored by git.
- The proposal PDF is stored in `doc/`.
