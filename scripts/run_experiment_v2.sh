#!/usr/bin/env bash
# Full v2 keypoint experiment — copy-paste on friend GPU machine.
# Does: git pull → preprocess v2 → train CEJ/intersection/apex → auto registry → git push logs
#
# Usage:
#   chmod +x scripts/run_experiment_v2.sh
#   ./scripts/run_experiment_v2.sh
#
# Optional env:
#   REPO=~/faraz/Test_work/research-work
#   RAW_ROOT=data/DenPAR/Dataset
#   BATCH=4

set -euo pipefail

REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
EXPERIMENT=v2
DATA_ROOT="data/processed_v2"
BATCH="${BATCH:-4}"
RAW_ROOT="${RAW_ROOT:-data/DenPAR/Dataset}"

echo "=== cd + git pull ==="
cd "$REPO"
git pull origin denpar-severity-replication

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  echo "ERROR: DenPAR not found at RAW_ROOT=$RAW_ROOT"
  echo "Set: RAW_ROOT=/path/to/DenPAR/Dataset ./scripts/run_experiment_v2.sh"
  exit 1
fi

echo "=== Preprocess $EXPERIMENT (strict bbox) ==="
python -m src.preprocess.prepare_dataset \
  --strategy v2 \
  --output-root "$DATA_ROOT" \
  --raw-root "$RAW_ROOT"

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
echo "Metrics registry: research_log/experiments/paper_table.json"
echo "If all 3 models finished, logs were auto-committed and pushed."
echo "On laptop: git pull"
