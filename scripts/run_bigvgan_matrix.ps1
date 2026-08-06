$ErrorActionPreference = "Stop"

$py = "..\bird_song_venv\Scripts\python.exe"
$seeds = @(42, 123, 777)
$root = "runs\experiments"

function Invoke-Checked {
    param([string[]]$Arguments)
    & $py -B @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

foreach ($seed in $seeds) {
    Invoke-Checked @(
        "scripts/04_train_wgan_gp.py",
        "--epochs", "20",
        "--batch-size", "32",
        "--critic-steps", "2",
        "--learning-rate", "1e-4",
        "--workers", "0",
        "--seed", "$seed",
        "--device", "cuda",
        "--output-dir", "$root/wgan_current_seed$seed",
        "--overwrite"
    )
    Invoke-Checked @(
        "scripts/04_train_wgan_gp.py",
        "--epochs", "20",
        "--batch-size", "32",
        "--critic-steps", "3",
        "--learning-rate", "5e-5",
        "--critic-learning-rate", "1e-4",
        "--workers", "0",
        "--seed", "$seed",
        "--device", "cuda",
        "--output-dir", "$root/wgan_stability_seed$seed",
        "--overwrite"
    )
}

foreach ($seed in $seeds) {
    Invoke-Checked @(
        "scripts/06_train_transformer.py",
        "--epochs", "60",
        "--batch-size", "32",
        "--workers", "0",
        "--seed", "$seed",
        "--device", "cuda",
        "--model-config", "configs/transformer_16x16.json",
        "--output-dir", "$root/transformer_16x16_seed$seed",
        "--overwrite"
    )
    Invoke-Checked @(
        "scripts/06_train_transformer.py",
        "--epochs", "60",
        "--batch-size", "32",
        "--workers", "0",
        "--seed", "$seed",
        "--device", "cuda",
        "--model-config", "configs/transformer_8x16.json",
        "--output-dir", "$root/transformer_8x16_seed$seed",
        "--overwrite"
    )
}

Write-Host "Completed WGAN and Transformer BigVGAN seed matrix."
