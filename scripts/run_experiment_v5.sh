#!/usr/bin/env bash
# Full v5 keypoint experiment — v4 CEJ/apex + bone-line endpoint intersections.
#
# v5 data  -> data/processed_v5/
# v5 weights -> runs/keypoints/v5_{cej,intersection,apex}/
#
# Usage:
#   bash scripts/run_experiment_v5.sh
#
# Optional:
#   SKIP_PREPROCESS=1
#   FORCE_REBUILD=1
#   BATCH=4

set -euo pipefail

REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
EXPERIMENT=v5
DATA_ROOT="data/processed_v5"
BATCH="${BATCH:-4}"
RAW_ROOT="${RAW_ROOT:-data/DenPAR/Dataset}"
GRACE_STEP_PX="${GRACE_STEP_PX:-1}"
MAX_GRACE_PX="${MAX_GRACE_PX:-8}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

echo "=== cd + git pull ==="
cd "$REPO"
git pull origin denpar-severity-replication

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== Folder safety ==="
echo "  v1/v2/v3/v4 data + older weights: NOT modified"
echo "  v5 data:  $DATA_ROOT/"
echo "  v5 weights: runs/keypoints/v5_*/"
echo "  v5 intersection: bone-line endpoints -> nearest tooth (no mask/ray)"
echo ""

if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  for candidate in "data/DenPAR/Dataset" "$REPO/data/DenPAR/Dataset"; do
    if [[ -d "$candidate/Training/Key Points Annotations" ]]; then
      RAW_ROOT="$candidate"
      break
    fi
  done
fi

if [[ "$FORCE_REBUILD" == "1" ]]; then
  rm -rf "$DATA_ROOT"
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  if [[ -d "$DATA_ROOT/keypoints/cej/train" ]]; then
    echo "SKIP preprocess: $DATA_ROOT exists (FORCE_REBUILD=1 to rebuild)"
  else
    if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
      echo "ERROR: Raw DenPAR not found."
      exit 1
    fi
    echo "=== Preprocess $EXPERIMENT -> $DATA_ROOT (v4 CEJ/apex + endpoint intersections) ==="
    python -m src.preprocess.prepare_dataset \
      --strategy v5 \
      --output-root "$DATA_ROOT" \
      --raw-root "$RAW_ROOT" \
      --grace-step-px "$GRACE_STEP_PX" \
      --max-grace-px "$MAX_GRACE_PX"
  fi
fi

if [[ ! -d "$DATA_ROOT/keypoints/cej/train" ]]; then
  echo "ERROR: Missing $DATA_ROOT — run preprocess with --strategy v5"
  exit 1
fi

echo "=== Train Keypoint R-CNN x3 ($EXPERIMENT) ==="
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
