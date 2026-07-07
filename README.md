# Bird Song Generation

Generative modeling project for bird song audio. The current repository structure focuses on a reproducible setup, dataset split manifests, and a small demo script. The audio dataset itself is expected to be available locally and is not committed to this repository.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── demo.py
├── doc/
│   └── Proposal_Generative Modeling of Toronto Bird Songs Using VAEs and Diffusion Models.pdf
├── manifests/
│   ├── full_dataset_manifest.csv
│   ├── full_dataset_train.csv
│   ├── full_dataset_validation.csv
│   └── full_dataset_test.csv
└── bird_songs_dataset/        # local only, not committed
    ├── bird_songs_metadata.csv
    └── wavfiles/
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
├── bird_songs_metadata.csv
└── wavfiles/
```

The checked-in manifest CSVs reference WAV files by relative path, for example `wavfiles/70119-0.wav`. This assumes each teammate has the same local dataset layout.

## Current Workflow

The project is organized around five planned stages:

1. Read the dataset and create train/validation/test splits.
2. Convert WAV files to spectrograms.
3. Train a supervised species classifier.
4. Train a VAE on spectrograms.
5. Train a diffusion model on spectrograms.

Only Step 1 has been prepared so far. The split manifests use all five species in the metadata and keep clips from the same source recording id in the same split to reduce leakage.

## Usage

Run the demo to verify that the local dataset and split manifests are readable:

```bash
python demo.py
```

Example output:

```text
Metadata rows: 5422
Manifest rows: 5422
Species: 5
Sample: Song Sparrow -> bird_songs_dataset/wavfiles/...
```

## Notes

- The audio dataset is not committed because it is large and should be shared separately.
- Generated spectrograms, checkpoints, model files, and run logs are ignored by git.
- The proposal PDF is stored in `doc/`.
