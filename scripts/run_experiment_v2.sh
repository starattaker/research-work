#!/usr/bin/env bash
# Full v2 keypoint experiment — friend GPU machine.
# SAFE: never writes to data/processed/ or runs/keypoints/{cej,intersection,apex}/
#
# v2 data  -> data/processed_v2/
# v2 weights -> runs/keypoints/v2_{cej,intersection,apex}/
#
# Usage:
#   bash scripts/run_experiment_v2.sh
#
# Optional env:
#   RAW_ROOT=data/DenPAR/Dataset   (raw DenPAR JSON — NOT data/processed)
#   BATCH=4
#   SKIP_PREPROCESS=1              (if data/processed_v2/ already built)

set -euo pipefail

REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
EXPERIMENT=v2
DATA_ROOT="data/processed_v2"
BATCH="${BATCH:-4}"
RAW_ROOT="${RAW_ROOT:-data/DenPAR/Dataset}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

echo "=== cd + git pull ==="
cd "$REPO"
git pull origin denpar-severity-replication

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== Folder safety (v1 stays untouched) ==="
echo "  v1 data:     data/processed/          (NOT modified)"
echo "  v1 weights:  runs/keypoints/cej|intersection|apex/  (NOT modified)"
echo "  v2 data:     $DATA_ROOT/               (new or re-built)"
echo "  v2 weights:  runs/keypoints/v2_*/       (separate best.pt per model)"
echo ""

# Auto-detect raw DenPAR if default path missing
if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
  for candidate in \
    "data/DenPAR/Dataset" \
    "$REPO/data/DenPAR/Dataset" \
    "../DenPAR/Dataset"; do
    if [[ -d "$candidate/Training/Key Points Annotations" ]]; then
      RAW_ROOT="$candidate"
      echo "Found raw DenPAR at: $RAW_ROOT"
      break
    fi
  done
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  if [[ -d "$DATA_ROOT/keypoints/cej/train" ]]; then
    echo "SKIP preprocess: $DATA_ROOT already exists (set SKIP_PREPROCESS=0 to force re-run)"
  else
    if [[ ! -d "$RAW_ROOT/Training/Key Points Annotations" ]]; then
      echo "ERROR: Raw DenPAR not found."
      echo "  data/processed/ is v1 OUTPUT — cannot be used as raw input for v2."
      echo "  You need the original DenPAR folder (Training/Key Points Annotations/...)."
      echo "  Download or set: RAW_ROOT=/path/to/DenPAR/Dataset bash scripts/run_experiment_v2.sh"
      exit 1
    fi
    echo "=== Preprocess $EXPERIMENT -> $DATA_ROOT (strict bbox) ==="
    python -m src.preprocess.prepare_dataset \
      --strategy v2 \
      --output-root "$DATA_ROOT" \
      --raw-root "$RAW_ROOT"
  fi
else
  echo "SKIP_PREPROCESS=1 — using existing $DATA_ROOT"
fi

if [[ ! -d "$DATA_ROOT/keypoints/cej/train" ]]; then
  echo "ERROR: Missing $DATA_ROOT/keypoints/cej/train — preprocess first."
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
echo "Metrics: research_log/experiments/paper_table.json"
echo "v1 weights still at runs/keypoints/cej/ etc."
