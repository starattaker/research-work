#!/usr/bin/env bash
# ONE command: sync + v6 ICC + axis severity + figures + push (production weights).
# Intersection: always v6 (OKS 0.894). v7 intersection excluded from ICC.
set -euo pipefail
cd "$(dirname "$0")/.."
SYNC_MODE=merge bash scripts/sync_friend_repo.sh
source venv/bin/activate
export PYTHONPATH=.
unset RAW_ROOT 2>/dev/null || true

CEJ_W="runs/keypoints/v6_cej/best.pt"
INT_W="runs/keypoints/v6_intersection/best.pt"
APEX_W="runs/keypoints/v6_apex/best.pt"

for w in "$CEJ_W" "$INT_W" "$APEX_W"; do
  if [[ ! -f "$w" ]]; then
    echo "ERROR: Missing $w — train v6 first (bash scripts/run_v6_experiment.sh)"
    exit 1
  fi
done

echo "=== 1/5 Production ICC parameter sweep (v6 all heads) ==="
python scripts/run_icc_parameter_sweep.py \
  --cej-weights "$CEJ_W" \
  --intersection-weights "$INT_W" \
  --apex-weights "$APEX_W" \
  --out research_log/icc_parameter_sweep.json

echo "=== 2/5 Axis-constrained severity ICC (v6; intersection required for Eq.1) ==="
python scripts/compare_axis_severity_icc.py \
  --split all \
  --cej-weights "$CEJ_W" \
  --intersection-weights "$INT_W" \
  --apex-weights "$APEX_W" \
  --out research_log/axis_severity_icc.json

echo "=== 3/5 Paper axis figures (multiple test images) ==="
python scripts/visualize_axis_severity_paper.py \
  --stems 431 5 100 240 622 18 358 530 76 903

echo "=== 4/5 Copy key sweep figures into paper/figures if present ==="
mkdir -p paper/figures
for src in \
  research_log/figures/grace_radius_sweep/grace_radius_sweep.png \
  research_log/figures/point_assignment_full/point_assignment_full.png \
  research_log/figures/apex_merge_analysis/apex_distance_hist_cdf.png
do
  if [[ -f "$src" ]]; then
    cp -f "$src" "paper/figures/$(basename "$src")"
    echo "  copied $src"
  fi
done

echo "=== 5/5 Push JSON + figures to GitHub ==="
bash scripts/push_research_figures.sh || echo "WARN: git push failed — commit locally"

echo ""
echo "DONE. Production ICC uses v6 intersection (OKS 0.894). v7 intersection NOT used."
echo "Outputs: research_log/icc_parameter_sweep.json"
echo "         research_log/axis_severity_icc.json"
echo "         paper/figures/"
