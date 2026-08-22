#!/usr/bin/env bash
# v5 intersection only: rebuild labels + train intersection Keypoint R-CNN.
#
# v5 = v4 CEJ/apex (region growing) + bone-line endpoints -> nearest TOOTH MASK.
# Only intersection labels/weights differ from v4; CEJ/apex unchanged.
#
# Compare after training:
#   v4 intersection OKS: runs/keypoints/v4_intersection/metrics.json
#   v5 intersection OKS: runs/keypoints/v5_intersection/metrics.json
#
# Usage (friend GPU):
#   bash scripts/run_v5_intersection.sh
#
# Optional:
#   BATCH=4
#   SKIP_PREPROCESS=1   # labels already rebuilt

set -euo pipefail

REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
DATA_ROOT="data/processed_v5"
BATCH="${BATCH:-4}"
RAW_ROOT="${RAW_ROOT:-data/DenPAR/Dataset}"
GRACE_STEP_PX="${GRACE_STEP_PX:-1}"
MAX_GRACE_PX="${MAX_GRACE_PX:-8}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

echo "=== cd + git pull ==="
cd "$REPO"
git pull origin denpar-severity-replication

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== v5 intersection experiment ==="
echo "  Labels:  $DATA_ROOT/keypoints/intersection/"
echo "  Weights: runs/keypoints/v5_intersection/"
echo "  Logic:   bone-line endpoints -> nearest tooth MASK (CEJ/apex = v4)"
echo ""

if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  for candidate in "data/DenPAR/Dataset" "$REPO/data/DenPAR/Dataset"; do
    if [[ -d "$candidate/Training/Key Points Annotations" ]]; then
      RAW_ROOT="$candidate"
      break
    fi
  done
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
    echo "ERROR: Raw DenPAR not found at $RAW_ROOT"
    exit 1
  fi
  echo "=== Rebuild v5 preprocess -> $DATA_ROOT ==="
  python -m src.preprocess.prepare_dataset \
    --strategy v5 \
    --output-root "$DATA_ROOT" \
    --raw-root "$RAW_ROOT" \
    --grace-step-px "$GRACE_STEP_PX" \
    --max-grace-px "$MAX_GRACE_PX"
else
  echo "SKIP preprocess (SKIP_PREPROCESS=1)"
fi

if [[ ! -d "$DATA_ROOT/keypoints/intersection/train" ]]; then
  echo "ERROR: Missing $DATA_ROOT/keypoints/intersection/train"
  exit 1
fi

echo "=== Train v5 intersection only ==="
python -m src.keypoint.train \
  --data-root "$DATA_ROOT/keypoints/intersection" \
  --keypoint-type intersection \
  --output-dir runs/keypoints/v5_intersection \
  --experiment-id v5 \
  --batch-size "$BATCH" \
  --patience 30 \
  --device cuda

echo ""
echo "=== Done ==="
echo "Compare: runs/keypoints/v4_intersection vs runs/keypoints/v5_intersection"
