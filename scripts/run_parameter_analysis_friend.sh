#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
source venv/bin/activate 2>/dev/null || true

RAW_ROOT="${RAW_ROOT:-data/DenPAR}"

if [[ ! -d "$RAW_ROOT" ]]; then
  echo "ERROR: RAW_ROOT not found: $RAW_ROOT"
  echo "Find DenPAR on disk:  find ~ -maxdepth 4 -type d -name 'Testing' 2>/dev/null | head"
  echo "Then: export RAW_ROOT=/path/to/parent  (parent contains Training/, Testing/, ...)"
  exit 1
fi

echo "Using RAW_ROOT=$RAW_ROOT"

echo "=== 1/3 Apex distance on double-root teeth (GT v6) ==="
python scripts/analyze_apex_merge_radius.py \
  --data-root data/processed_v6 \
  --split all

echo ""
echo "=== 2/3 Grace radius sweep (raw DenPAR) ==="
python scripts/analyze_grace_radius_sweep.py \
  --raw-root "$RAW_ROOT" \
  --split Testing \
  --max-radius 24

echo ""
echo "=== 3/3 Slot-axis ICC comparison ==="
bash scripts/run_slot_axis_compare_friend.sh

echo ""
echo "Outputs:"
echo "  research_log/figures/apex_merge_analysis/"
echo "  research_log/figures/grace_radius_sweep/"
echo "  research_log/slot_axis_icc_comparison.json"
echo "  research_log/figures/slot_axis_methods/"
