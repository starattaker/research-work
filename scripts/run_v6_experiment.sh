#!/usr/bin/env bash
# Full v6 keypoint experiment — PCA-axis CEJ/apex slots + endpoint intersections (L/R bone lines).
#
# v6 data  -> data/processed_v6/
# v6 weights -> runs/keypoints/v6_{cej,intersection,apex}/
#
# Compare after training:
#   python -c "
#   import json; from pathlib import Path
#   for run in ('v4_intersection','v5_intersection','v6_intersection'):
#       p=Path(f'runs/keypoints/{run}/metrics.json')
#       if p.exists(): print(run, round(json.loads(p.read_text())['test_oks'],4))
#   "
#
# Usage (friend GPU — paste as one block):
#   cd ~/faraz/Test_work/research-work && git pull origin denpar-severity-replication && rm -rf data/processed_v6 && bash scripts/run_v6_experiment.sh
#
# Optional:
#   BATCH=4
#   SKIP_PREPROCESS=1

set -euo pipefail

REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
EXPERIMENT=v6
DATA_ROOT="data/processed_v6"
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
echo "=== v6 experiment ==="
echo "  Labels:  $DATA_ROOT/"
echo "  Weights: runs/keypoints/v6_*/"
echo "  CEJ/apex: region grow + PCA long-axis slots (no x-sort pad)"
echo "  Intersection: endpoints -> nearest mask, max 2 (left/right bone line)"
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
  echo "=== Rebuild v6 preprocess -> $DATA_ROOT ==="
  python -m src.preprocess.prepare_dataset \
    --strategy v6 \
    --output-root "$DATA_ROOT" \
    --raw-root "$RAW_ROOT" \
    --grace-step-px "$GRACE_STEP_PX" \
    --max-grace-px "$MAX_GRACE_PX"
else
  echo "SKIP preprocess (SKIP_PREPROCESS=1)"
fi

if [[ ! -d "$DATA_ROOT/keypoints/cej/train" ]]; then
  echo "ERROR: Missing $DATA_ROOT/keypoints/cej/train"
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
echo "Compare intersection OKS: v4 vs v5 vs v6 in runs/keypoints/*/metrics.json"
