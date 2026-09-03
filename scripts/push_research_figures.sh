#!/usr/bin/env bash
# Push generated figures from friend GPU to GitHub (run on Pop!_OS after sweeps).
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-denpar-severity-replication}"
MSG="${MSG:-Add research figures from GPU runs}"

DIRS=(
  research_log/figures/point_assignment_full
  research_log/figures/grace_radius_sweep
  research_log/figures/apex_merge_analysis
  research_log/figures/region_growing
  research_log/figures/axis_severity
  paper/figures
)

echo "=== Staging figure directories ==="
for d in "${DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    git add "$d"/*.png "$d"/*.json "$d"/*.gif 2>/dev/null || true
    git add "$d" 2>/dev/null || true
    echo "  + $d"
  else
    echo "  skip (missing): $d"
  fi
done

git add research_log/icc_*.json research_log/axis_severity_icc.json 2>/dev/null || true

if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git status --short
git commit -m "$MSG"
git push origin "$BRANCH"
echo "Pushed figures to origin/$BRANCH"
