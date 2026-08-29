#!/usr/bin/env bash
# Friend GPU: regenerate step 1-5 viz figures (no GitHub weight upload).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

source venv/bin/activate 2>/dev/null || true

python scripts/visualize_severity_pipeline_steps.py \
  --yolo-weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt \
  --cej-weights runs/keypoints/v6_cej/best.pt \
  --intersection-weights runs/keypoints/v6_intersection/best.pt \
  --apex-weights runs/keypoints/v6_apex/best.pt \
  --data-root data/processed_v6 \
  --split test \
  --n-images 5 \
  --seed 42 \
  --inference-mode full \
  "$@"

echo ""
echo "Figures: research_log/figures/severity_pipeline_steps/"
echo "Compare step4: G=keypoints vs P=predictions"
echo "Try --gt-proposals to remove YOLO box shift from step 4"
