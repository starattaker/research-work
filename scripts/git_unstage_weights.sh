#!/usr/bin/env bash
# Remove failed weight upload from git history (run on friend if push was rejected).
set -euo pipefail
cd "$(dirname "$0")/.."
git rm -r --cached artifacts/local_viz/*.pt 2>/dev/null || true
git reset HEAD artifacts/local_viz/*.pt 2>/dev/null || true
echo "Weights unstaged. Commit code-only changes only."
