#!/usr/bin/env bash
# ONE command: sync + grace-radius sweep (0–48px) + bbox outlier analysis for v7 preprocess.
# Graphs: research_log/figures/point_assignment_full/
# Also re-runs legacy 0–24px grace sweep for comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
SYNC_MODE=merge bash scripts/sync_friend_repo.sh
source venv/bin/activate
export PYTHONPATH=.

DEFAULT_RAW="data/DenPAR/Dataset"
if [[ -z "${RAW_ROOT:-}" ]] || [[ ! -d "${RAW_ROOT}/Testing/Key Points Annotations" ]]; then
  RAW_ROOT="$DEFAULT_RAW"
fi
if [[ ! -d "$RAW_ROOT/Testing/Key Points Annotations" ]]; then
  echo "ERROR: DenPAR not at $RAW_ROOT"
  exit 1
fi
echo "Using RAW_ROOT=$RAW_ROOT"

echo "=== 1/2 Full assignment sweep (0–48px, all splits) ==="
python scripts/analyze_point_assignment_full.py \
  --raw-root "$RAW_ROOT" \
  --split all \
  --max-radius "${MAX_RADIUS:-48}"

echo ""
echo "=== 2/2 Legacy grace sweep (0–24px, test only — prior graph) ==="
python scripts/analyze_grace_radius_sweep.py \
  --raw-root "$RAW_ROOT" \
  --split Testing \
  --max-radius 24

echo ""
echo "=== Apex merge distance (GT double-root) ==="
python scripts/analyze_apex_merge_radius.py \
  --data-root data/processed_v6 \
  --split all

echo ""
echo "Outputs:"
echo "  research_log/figures/point_assignment_full/point_assignment_full.png  ← main graph"
echo "  research_log/figures/point_assignment_full/point_assignment_report.json"
echo "  research_log/point_assignment_report.json  ← training picks up max_grace_px here"
echo "  research_log/figures/grace_radius_sweep/grace_radius_sweep.png  ← 0–24px comparison"
echo "  research_log/figures/apex_merge_analysis/apex_distance_hist_cdf.png"
