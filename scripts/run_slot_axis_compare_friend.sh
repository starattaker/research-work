#!/usr/bin/env bash
# Friend GPU: compare ICC for 3 slot-axis methods + save PCA viz figures.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
source venv/bin/activate 2>/dev/null || true

DEFAULT_RAW="data/DenPAR/Dataset"
if [[ -z "${RAW_ROOT:-}" ]] || [[ ! -d "${RAW_ROOT}/Testing/Key Points Annotations" ]]; then
  if [[ -n "${RAW_ROOT:-}" ]]; then
    echo "WARN: RAW_ROOT='$RAW_ROOT' not valid; using $DEFAULT_RAW"
  fi
  RAW_ROOT="$DEFAULT_RAW"
fi

echo "Using RAW_ROOT=$RAW_ROOT"

python scripts/compare_slot_axis_icc.py \
  --yolo-weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt \
  --cej-weights runs/keypoints/v6_cej/best.pt \
  --intersection-weights runs/keypoints/v6_intersection/best.pt \
  --apex-weights runs/keypoints/v6_apex/best.pt \
  --data-root data/processed_v6 \
  --raw-root "$RAW_ROOT" \
  --split test \
  "$@"

python scripts/visualize_slot_axis_methods.py \
  --yolo-weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt \
  --cej-weights runs/keypoints/v6_cej/best.pt \
  --intersection-weights runs/keypoints/v6_intersection/best.pt \
  --apex-weights runs/keypoints/v6_apex/best.pt \
  --data-root data/processed_v6 \
  --raw-root "$RAW_ROOT" \
  --split test \
  --n-images 5 \
  --seed 42

echo ""
echo "ICC report: research_log/slot_axis_icc_comparison.json"
echo "PCA figures: research_log/figures/slot_axis_methods/"
