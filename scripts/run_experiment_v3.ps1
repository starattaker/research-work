# Full v3 keypoint experiment on Windows (after v3 preprocess finishes).
# Usage (PowerShell):
#   cd c:\Oralvis_Seekright
#   .\scripts\run_experiment_v3.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

git pull origin denpar-severity-replication

$venvPython = Join-Path $PWD "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv not found. Run once:"
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}

$env:PYTHONPATH = "."
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$EXPERIMENT = "v3"
$DATA_ROOT = "data/processed_v3"
$BATCH = if ($env:BATCH) { $env:BATCH } else { 4 }

if (-not (Test-Path "$DATA_ROOT/keypoints/cej/train")) {
    Write-Host "ERROR: v3 data missing at $DATA_ROOT"
    Write-Host "Run preprocess first:"
    Write-Host "  .\venv\Scripts\python.exe -m src.preprocess.prepare_dataset --strategy v3 --output-root data/processed_v3"
    exit 1
}

Write-Host "=== Train Keypoint R-CNN x3 ($EXPERIMENT) ==="
foreach ($KPT in @("cej", "intersection", "apex")) {
    Write-Host "--- $EXPERIMENT / $KPT ---"
    & $venvPython -m src.keypoint.train `
        --data-root "$DATA_ROOT/keypoints/$KPT" `
        --keypoint-type $KPT `
        --output-dir "runs/keypoints/${EXPERIMENT}_$KPT" `
        --experiment-id $EXPERIMENT `
        --batch-size $BATCH `
        --patience 30 `
        --device cuda
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "=== Done: $EXPERIMENT ==="
Write-Host "Registry: research_log/experiments/paper_table.json"
