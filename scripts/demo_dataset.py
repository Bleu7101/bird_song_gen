from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "bird_songs_dataset"
METADATA_CSV = DATASET_DIR / "bird_songs_metadata.csv"
MANIFEST_CSV = PROJECT_ROOT / "manifests" / "full_dataset_manifest.csv"


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_CSV}. "
            "Place the dataset at bird_songs_dataset/ before running the demo."
        )
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Missing manifest file: {MANIFEST_CSV}")

    metadata = pd.read_csv(METADATA_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    sample = manifest.iloc[0]
    sample_wav = DATASET_DIR / sample["relative_wav_path"]

    print(f"Metadata rows: {len(metadata)}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Species: {manifest['name'].nunique()}")
    print(f"Splits: {', '.join(sorted(manifest['split'].unique()))}")
    print(f"Sample: {sample['name']} -> {sample_wav}")
    print(f"Sample WAV exists locally: {sample_wav.exists()}")


if __name__ == "__main__":
    main()
