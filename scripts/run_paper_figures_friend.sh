#!/usr/bin/env bash
# Friend GPU: axis severity ICC + paper figures + push to GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
SYNC_MODE=merge bash scripts/sync_friend_repo.sh
source venv/bin/activate
export PYTHONPATH=.

echo "=== 1/3 Axis severity ICC (v6 weights) ==="
python scripts/compare_axis_severity_icc.py \
  --split all \
  --out research_log/axis_severity_icc.json

echo "=== 2/3 Paper axis figures (5 test images) ==="
python scripts/visualize_axis_severity_paper.py \
  --stems 431 5 100 240 622 18 358 530

echo "=== 3/3 Optional v7 ICC ==="
if [[ -f runs/keypoints/v7_cej/best.pt ]]; then
  python scripts/run_icc_parameter_sweep.py \
    --cej-weights runs/keypoints/v7_cej/best.pt \
    --intersection-weights runs/keypoints/v7_intersection/best.pt \
    --apex-weights runs/keypoints/v7_apex/best.pt \
    --out research_log/icc_v7_report.json || true
fi

echo "=== Push figures (set GIT credentials if push fails) ==="
bash scripts/push_research_figures.sh || echo "WARN: git push failed — commit locally and push manually"
