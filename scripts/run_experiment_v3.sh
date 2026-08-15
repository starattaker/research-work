#!/usr/bin/env bash
# Full v3 keypoint experiment — same as v2 but mask+4px grace preprocessing.
#
# Usage:
#   chmod +x scripts/run_experiment_v3.sh
#   ./scripts/run_experiment_v3.sh

set -euo pipefail

REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
EXPERIMENT=v3
DATA_ROOT="data/processed_v3"
BATCH="${BATCH:-4}"
RAW_ROOT="${RAW_ROOT:-data/DenPAR/Dataset}"
GRACE_PX="${GRACE_PX:-4}"

echo "=== cd + git pull ==="
cd "$REPO"
git pull origin denpar-severity-replication

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  echo "ERROR: DenPAR not found at RAW_ROOT=$RAW_ROOT"
  exit 1
fi

echo "=== Preprocess $EXPERIMENT (mask + ${GRACE_PX}px grace) ==="
python -m src.preprocess.prepare_dataset \
  --strategy v3 \
  --output-root "$DATA_ROOT" \
  --raw-root "$RAW_ROOT" \
  --grace-px "$GRACE_PX"

echo "=== Optional: preprocessing stats ==="
python scripts/compare_preprocessing.py --raw-root "$RAW_ROOT" || true

echo "=== Train Keypoint R-CNN ×3 ($EXPERIMENT) ==="
for KPT in cej intersection apex; do
  echo "--- $EXPERIMENT / $KPT ---"
  python -m src.keypoint.train \
    --data-root "$DATA_ROOT/keypoints/$KPT" \
    --keypoint-type "$KPT" \
    --output-dir "runs/keypoints/${EXPERIMENT}_${KPT}" \
    --experiment-id "$EXPERIMENT" \
    --batch-size "$BATCH" \
    --patience 30 \
    --device cuda
done

echo ""
echo "=== Done: $EXPERIMENT ==="
echo "Registry: research_log/experiments/paper_table.json"
