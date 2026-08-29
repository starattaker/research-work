#!/usr/bin/env bash
# Friend GPU: exhaustive ICC diagnosis (run before changing pipeline code).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

source venv/bin/activate 2>/dev/null || true

python scripts/diagnose_severity_icc.py \
  --yolo-weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt \
  --cej-weights runs/keypoints/v6_cej/best.pt \
  --intersection-weights runs/keypoints/v6_intersection/best.pt \
  --apex-weights runs/keypoints/v6_apex/best.pt \
  --data-root data/processed_v6 \
  --split test \
  "$@"

echo ""
echo "Report: research_log/severity_icc_diagnosis.json"
echo "Read diagnosis[] bullets + compare D1 vs D2 vs D3 vs H1-H3"
