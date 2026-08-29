# Unpack local_viz_bundle.tar.gz into this repo (weights + 5-image data).
param(
    [string]$Archive = "local_viz_bundle.tar.gz",
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent)
)

Set-Location $RepoRoot

if (-not (Test-Path $Archive)) {
    Write-Error "Archive not found: $(Join-Path $RepoRoot $Archive)"
}

tar -xzf $Archive

$bundle = Join-Path $RepoRoot "local_viz_bundle"
New-Item -ItemType Directory -Force -Path runs/detect, runs/keypoints/v6_cej, runs/keypoints/v6_intersection, runs/keypoints/v6_apex | Out-Null
Copy-Item "$bundle/runs/detect/best.pt" runs/detect/yolov8x_tooth_best.pt -Force
Copy-Item "$bundle/runs/keypoints/v6_cej_best.pt" runs/keypoints/v6_cej/best.pt -Force
Copy-Item "$bundle/runs/keypoints/v6_intersection_best.pt" runs/keypoints/v6_intersection/best.pt -Force
Copy-Item "$bundle/runs/keypoints/v6_apex_best.pt" runs/keypoints/v6_apex/best.pt -Force

$dataDest = Join-Path $RepoRoot "data/processed_v6"
New-Item -ItemType Directory -Force -Path $dataDest | Out-Null
Copy-Item "$bundle/data/processed_v6/*" $dataDest -Recurse -Force

Write-Host "Unpacked to $RepoRoot"
Write-Host ""
Write-Host '  $env:PYTHONPATH="."'
Write-Host "  python scripts/visualize_severity_pipeline_steps.py ``"
Write-Host "    --yolo-weights runs/detect/yolov8x_tooth_best.pt ``"
Write-Host "    --cej-weights runs/keypoints/v6_cej/best.pt ``"
Write-Host "    --intersection-weights runs/keypoints/v6_intersection/best.pt ``"
Write-Host "    --apex-weights runs/keypoints/v6_apex/best.pt ``"
Write-Host "    --data-root data/processed_v6 --split test --n-images 5 --seed 42"
