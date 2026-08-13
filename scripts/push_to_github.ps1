# Push branch denpar-severity-replication to private repo research-work
# Run once: gh auth login

$ErrorActionPreference = "Stop"
$gh = if (Get-Command gh -ErrorAction SilentlyContinue) { "gh" } else { "$env:TEMP\ghcli\bin\gh.exe" }

Set-Location $PSScriptRoot\..

& $gh auth status
if ($LASTEXITCODE -ne 0) { Write-Host "Run: gh auth login"; exit 1 }

if (-not (Test-Path .git)) {
    git init
    git checkout -b denpar-severity-replication
    git add .
    git commit -m "DenPAR bone loss severity replication"
}

$remotes = git remote 2>$null
if ($remotes -notcontains "origin") {
    & $gh repo create research-work --private --source=. --remote=origin --description "DenPAR alveolar bone loss severity replication"
}

git push -u origin denpar-severity-replication
Write-Host "`nClone on Linux:"
Write-Host "  git clone -b denpar-severity-replication https://github.com/$(gh api user -q .login)/research-work.git"
