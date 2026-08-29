# Pull local_viz_bundle.tar.gz from friend GPU, then unpack for local viz.
param(
    [Parameter(Mandatory = $true)]
    [string]$GpuHost,
    [string]$RemoteRepo = "~/faraz/Test_work/research-work",
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent)
)

Set-Location $RepoRoot
$remote = "${GpuHost}:${RemoteRepo}/local_viz_bundle.tar.gz"

Write-Host "Pulling $remote ..."
scp $remote .

if ($LASTEXITCODE -ne 0) {
    Write-Error "scp failed. On GPU first run: bash scripts/pack_local_viz_bundle.sh"
}

& (Join-Path $PSScriptRoot "unpack_local_viz_bundle.ps1") -RepoRoot $RepoRoot
