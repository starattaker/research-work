# Full v3 keypoint experiment on Windows (after v3 preprocess finishes).
# Usage (PowerShell):
#   cd c:\Oralvis_Seekright
#   .\scripts\run_experiment_v3.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

git pull origin denpar-severity-replication

$env:PYTHONPATH = "."
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$EXPERIMENT = "v3"
$DATA_ROOT = "data/processed_v3"
$BATCH = if ($env:BATCH) { $env:BATCH } else { 4 }

Write-Host "=== Train Keypoint R-CNN x3 ($EXPERIMENT) ==="
foreach ($KPT in @("cej", "intersection", "apex")) {
    Write-Host "--- $EXPERIMENT / $KPT ---"
    python -m src.keypoint.train `
        --data-root "$DATA_ROOT/keypoints/$KPT" `
        --keypoint-type $KPT `
        --output-dir "runs/keypoints/${EXPERIMENT}_$KPT" `
        --experiment-id $EXPERIMENT `
        --batch-size $BATCH `
        --patience 30 `
        --device cuda
}

Write-Host ""
Write-Host "=== Done: $EXPERIMENT ==="
Write-Host "Registry: research_log/experiments/paper_table.json"
Write-Host "If all 3 models finished, logs were auto-committed (git push may need network)."
