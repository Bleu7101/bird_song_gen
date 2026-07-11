param(
    [string]$PythonVersion = "3.14",
    [string]$CudaWheel = "cu128"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path (Split-Path -Parent $ProjectRoot) "bird_song_venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

& py "-$PythonVersion" -c "import sys; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion is not available through the py launcher. Install it first, then rerun this script."
}

if (-not (Test-Path -LiteralPath $Python)) {
    & py "-$PythonVersion" -m venv $VenvDir
}

& $Python -m pip install --upgrade pip
& $Python -m pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CudaWheel"
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $Python -m pip install --editable $ProjectRoot --no-deps
& $Python (Join-Path $ProjectRoot "scripts\validate_setup.py")

Write-Host "Environment ready: $VenvDir"
Write-Host "Activate with: $VenvDir\Scripts\Activate.ps1"
