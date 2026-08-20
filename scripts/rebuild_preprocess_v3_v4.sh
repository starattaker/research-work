#!/usr/bin/env bash
# Delete old v3/v4 processed folders and rebuild with latest rules:
#   v3 = mask + 4px grace, tie-break mask centroid
#   v4 = region growing 1..8px rings, tie-break mask centroid
#
# Does NOT touch data/processed/ (v1) or data/processed_v2/ (v2).
#
# Usage:
#   bash scripts/rebuild_preprocess_v3_v4.sh
#
# Optional:
#   DELETE_RUNS=1     also remove runs/keypoints/v3_* and v4_*
#   RAW_ROOT=...      path to DenPAR/Dataset

set -euo pipefail

REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
RAW_ROOT="${RAW_ROOT:-data/DenPAR/Dataset}"
GRACE_PX="${GRACE_PX:-4}"
GRACE_STEP_PX="${GRACE_STEP_PX:-1}"
MAX_GRACE_PX="${MAX_GRACE_PX:-8}"
DELETE_RUNS="${DELETE_RUNS:-0}"

echo "=== cd + git pull ==="
cd "$REPO"
git pull origin denpar-severity-replication

export PYTHONPATH=.

if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  for candidate in "data/DenPAR/Dataset" "$REPO/data/DenPAR/Dataset"; do
    if [[ -d "$candidate/Training/Key Points Annotations" ]]; then
      RAW_ROOT="$candidate"
      break
    fi
  done
fi

if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  echo "ERROR: Raw DenPAR not found. Set RAW_ROOT=/path/to/DenPAR/Dataset"
  exit 1
fi

echo ""
echo "=== Safety ==="
echo "  KEEP: data/processed/ (v1), data/processed_v2/ (v2)"
echo "  DELETE + REBUILD: data/processed_v3/, data/processed_v4/"
if [[ "$DELETE_RUNS" == "1" ]]; then
  echo "  DELETE: runs/keypoints/v3_*, runs/keypoints/v4_*"
fi
echo ""

rm -rf data/processed_v3 data/processed_v4
if [[ "$DELETE_RUNS" == "1" ]]; then
  rm -rf runs/keypoints/v3_cej runs/keypoints/v3_intersection runs/keypoints/v3_apex
  rm -rf runs/keypoints/v4_cej runs/keypoints/v4_intersection runs/keypoints/v4_apex
fi

echo "=== Preprocess v3 -> data/processed_v3 (mask + ${GRACE_PX}px, mask centroid tie-break) ==="
python -m src.preprocess.prepare_dataset \
  --strategy v3 \
  --output-root data/processed_v3 \
  --raw-root "$RAW_ROOT" \
  --grace-px "$GRACE_PX"

echo ""
echo "=== Preprocess v4 -> data/processed_v4 (region grow 1-${MAX_GRACE_PX}px) ==="
python -m src.preprocess.prepare_dataset \
  --strategy v4 \
  --output-root data/processed_v4 \
  --raw-root "$RAW_ROOT" \
  --grace-step-px "$GRACE_STEP_PX" \
  --max-grace-px "$MAX_GRACE_PX"

echo ""
echo "=== Comparison table (v1/v2/v3/v4 stats) ==="
python scripts/compare_preprocessing.py --raw-root "$RAW_ROOT" \
  --grace-px "$GRACE_PX" \
  --grace-step-px "$GRACE_STEP_PX" \
  --max-grace-px "$MAX_GRACE_PX"

echo ""
echo "=== Done ==="
echo "  v3 data: data/processed_v3/"
echo "  v4 data: data/processed_v4/"
echo "  Stats:   research_log/preprocessing_comparison.md"
echo ""
echo "Train next (friend GPU):"
echo "  bash scripts/run_experiment_v3.sh"
echo "  bash scripts/run_experiment_v4.sh"
