#!/usr/bin/env bash
# Train v7 after point_assignment_report.json exists (run run_point_inclusion_friend.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/sync_friend_repo.sh
source venv/bin/activate
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT=v7
DATA_ROOT="data/processed_v7"
BATCH="${BATCH:-4}"
RAW_ROOT="${RAW_ROOT:-data/DenPAR/Dataset}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

# Defaults if sweep not run yet
MAX_GRACE_PX="${MAX_GRACE_PX:-12}"
BBOX_OUTLIER_PX="${BBOX_OUTLIER_PX:-24}"
CFG="research_log/point_assignment_report.json"
if [[ -f "$CFG" ]]; then
  echo "Reading preprocess knobs from $CFG"
  MAX_GRACE_PX=$(python -c "import json; c=json.load(open('$CFG')); print(int(c.get('max_grace_px', $MAX_GRACE_PX)))")
  BBOX_OUTLIER_PX=$(python -c "import json; c=json.load(open('$CFG')); print(float(c.get('bbox_outlier_margin_px', $BBOX_OUTLIER_PX)))")
fi

if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  for candidate in "data/DenPAR/Dataset" "$PWD/data/DenPAR/Dataset"; do
    if [[ -d "$candidate/Training/Key Points Annotations" ]]; then
      RAW_ROOT="$candidate"
      break
    fi
  done
fi

echo "=== v7 train ==="
echo "  DATA_ROOT=$DATA_ROOT"
echo "  MAX_GRACE_PX=$MAX_GRACE_PX  BBOX_OUTLIER_PX=$BBOX_OUTLIER_PX"
echo "  Weights: runs/keypoints/v7_*/"
echo ""

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
    echo "ERROR: Raw DenPAR not found at $RAW_ROOT"
    exit 1
  fi
  echo "=== Rebuild v7 preprocess ==="
  python -m src.preprocess.prepare_dataset \
    --strategy v6 \
    --output-root "$DATA_ROOT" \
    --raw-root "$RAW_ROOT" \
    --grace-step-px 1 \
    --max-grace-px "$MAX_GRACE_PX" \
    --bbox-outlier-margin-px "$BBOX_OUTLIER_PX"
else
  echo "SKIP preprocess (SKIP_PREPROCESS=1)"
fi

for KPT in cej intersection apex; do
  OUT="runs/keypoints/${EXPERIMENT}_${KPT}"
  echo "--- $EXPERIMENT / $KPT -> $OUT ---"
  python -m src.keypoint.train \
    --data-root "$DATA_ROOT/keypoints/$KPT" \
    --keypoint-type "$KPT" \
    --output-dir "$OUT" \
    --experiment-id "$EXPERIMENT" \
    --batch-size "$BATCH" \
    --patience 30 \
    --device cuda
done

echo ""
echo "=== Done: $EXPERIMENT ==="
echo "Re-run ICC: bash scripts/run_icc_optimize_friend.sh"
echo "  (update weights paths to runs/keypoints/v7_* in sweep script if needed)"
