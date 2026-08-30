#!/usr/bin/env bash
# Friend GPU: step-by-step slot-axis figures + oracle EDA.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
source venv/bin/activate 2>/dev/null || true

DEFAULT_RAW="data/DenPAR/Dataset"
if [[ -z "${RAW_ROOT:-}" ]] || [[ ! -d "${RAW_ROOT}/Testing/Key Points Annotations" ]]; then
  RAW_ROOT="$DEFAULT_RAW"
fi
echo "Using RAW_ROOT=$RAW_ROOT"

WEIGHTS=(
  --yolo-weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt
  --cej-weights runs/keypoints/v6_cej/best.pt
  --intersection-weights runs/keypoints/v6_intersection/best.pt
  --apex-weights runs/keypoints/v6_apex/best.pt
)

echo "=== Step-by-step figures (5 images) ==="
python scripts/visualize_slot_axis_steps.py \
  "${WEIGHTS[@]}" \
  --data-root data/processed_v6 \
  --raw-root "$RAW_ROOT" \
  --split test \
  --n-images 5 \
  --seed 42

echo ""
echo "=== Oracle / mask-PCA EDA ==="
python scripts/oracle_slot_eda.py \
  "${WEIGHTS[@]}" \
  --data-root data/processed_v6 \
  --raw-root "$RAW_ROOT" \
  --split test

echo ""
echo "Outputs:"
echo "  research_log/figures/slot_axis_steps/step1..step6/"
echo "  research_log/oracle_slot_eda/oracle_slot_eda.json"
echo "  research_log/oracle_slot_eda/icc_by_method.png"
echo "  mask_available: research_log/slot_axis_icc_comparison.json (field mask_available)"
