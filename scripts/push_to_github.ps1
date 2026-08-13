# Create private GitHub repo "research-work" and push branch denpar-severity-replication
# Prerequisite: gh auth login  (run once interactively)

$ErrorActionPreference = "Stop"
$gh = if (Get-Command gh -ErrorAction SilentlyContinue) { "gh" } else { "$env:TEMP\ghcli\bin\gh.exe" }

Write-Host "Checking gh auth..."
& $gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run: gh auth login"
    exit 1
}

Set-Location $PSScriptRoot\..

if (-not (Test-Path .git)) {
    git init
    git branch -M denpar-severity-replication
}

git add .
git status

$commit = git log -1 --oneline 2>$null
if (-not $commit) {
    git commit -m "DenPAR bone loss severity replication — code, processed data, research log"
}

$repoExists = & $gh repo view YOUR_USERNAME/research-work 2>$null
if ($LASTEXITCODE -ne 0) {
    & $gh repo create research-work --private --source=. --remote=origin --description "DenPAR alveolar bone loss severity replication (research)"
}

git push -u origin denpar-severity-replication

Write-Host "Done. Clone with:"
Write-Host "  git clone -b denpar-severity-replication git@github.com:YOUR_USERNAME/research-work.git"
