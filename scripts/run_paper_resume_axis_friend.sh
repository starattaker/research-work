#!/usr/bin/env bash
# Resume after ICC sweep already wrote icc_parameter_sweep.json.
# Pulls latest (SideDetail cej fix), then axis ICC + figures + push.
set -euo pipefail
cd "$(dirname "$0")/.."
SYNC_MODE=merge bash scripts/sync_friend_repo.sh
source venv/bin/activate
export PYTHONPATH=.

CEJ_W="runs/keypoints/v6_cej/best.pt"
INT_W="runs/keypoints/v6_intersection/best.pt"
APEX_W="runs/keypoints/v6_apex/best.pt"

echo "=== Resume: axis severity ICC (skip GPU ICC sweep) ==="
python scripts/compare_axis_severity_icc.py \
  --split all \
  --cej-weights "$CEJ_W" \
  --intersection-weights "$INT_W" \
  --apex-weights "$APEX_W" \
  --out research_log/axis_severity_icc.json

echo "=== Paper axis figures ==="
python scripts/visualize_axis_severity_paper.py \
  --stems 431 5 100 240 622 18 358 530 76 903

mkdir -p paper/figures
for src in \
  research_log/figures/grace_radius_sweep/grace_radius_sweep.png \
  research_log/figures/point_assignment_full/point_assignment_full.png \
  research_log/figures/apex_merge_analysis/apex_distance_hist_cdf.png
do
  [[ -f "$src" ]] && cp -f "$src" "paper/figures/$(basename "$src")" && echo "  copied $src"
done

bash scripts/push_research_figures.sh || echo "WARN: git push failed"
echo "DONE. Outputs: research_log/axis_severity_icc.json  paper/figures/"
