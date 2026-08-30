#!/usr/bin/env bash
# Friend GPU: pull latest, then run full ICC pipeline (diagnose → train/val/test → sweep).
# Usage:  bash scripts/run_icc_friend_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
unset RAW_ROOT 2>/dev/null || true

source venv/bin/activate

YOLO="${YOLO:-runs/detect/runs/detection/yolov8x_tooth/weights/best.pt}"
CEJ="${CEJ:-runs/keypoints/v6_cej/best.pt}"
INT="${INT:-runs/keypoints/v6_intersection/best.pt}"
APEX="${APEX:-runs/keypoints/v6_apex/best.pt}"
DATA="${DATA:-data/processed_v6}"
DEVICE="${DEVICE:-cuda}"

echo "========== [1/5] GT sanity (no GPU models) =========="
python scripts/icc_gt_sanity.py --data-root "$DATA" --split test \
  --out research_log/icc_gt_sanity_test.json

echo ""
echo "========== [2/5] ICC diagnosis (tensor vs paper_x vs oracle) =========="
python scripts/diagnose_severity_icc.py \
  --data-root "$DATA" --split test --device "$DEVICE" \
  --yolo-weights "$YOLO" \
  --cej-weights "$CEJ" \
  --intersection-weights "$INT" \
  --apex-weights "$APEX" \
  --inference-mode roi \
  --out research_log/severity_icc_diagnosis.json

echo ""
echo "========== [3/5] ICC train / val / test (paper: roi + paper_x + both_sides) =========="
for split in train val test; do
  echo "--- ICC $split ---"
  python scripts/run_severity_icc.py \
    --data-root "$DATA" --split "$split" --device "$DEVICE" \
    --yolo-weights "$YOLO" \
    --cej-weights "$CEJ" \
    --intersection-weights "$INT" \
    --apex-weights "$APEX" \
    --inference-mode roi \
    --combine-mode paper_x \
    --gt-slot-convention paper_x \
    --severity-protocol both_sides \
    --out "research_log/severity_icc_${split}.json"
done

echo ""
echo "========== [4/5] Combine-mode sweep (test split) =========="
python scripts/sweep_icc_combine.py \
  --data-root "$DATA" --split test --device "$DEVICE" \
  --yolo-weights "$YOLO" \
  --cej-weights "$CEJ" \
  --intersection-weights "$INT" \
  --apex-weights "$APEX" \
  --out research_log/icc_combine_sweep.json

echo ""
echo "========== [5/5] Slot-axis comparison (mask_pca vs paper_x) =========="
python scripts/compare_slot_axis_icc.py \
  --data-root "$DATA" --split test --device "$DEVICE" \
  --yolo-weights "$YOLO" \
  --cej-weights "$CEJ" \
  --intersection-weights "$INT" \
  --apex-weights "$APEX" \
  --out research_log/slot_axis_icc_comparison.json

echo ""
echo "========== SUMMARY =========="
python - <<'PY'
import json
from pathlib import Path

print("End-to-end ICC (paper_x, both_sides, roi):")
for split in ("train", "val", "test"):
    p = Path(f"research_log/severity_icc_{split}.json")
    if p.exists():
        r = json.loads(p.read_text())
        icc = r.get("icc")
        print(f"  {split:5s}  ICC={icc:.4f}  n={r.get('n_pairs')}  (paper test target 0.801)" if icc else f"  {split}: n/a")

sp = Path("research_log/icc_combine_sweep.json")
if sp.exists():
    r = json.loads(sp.read_text())
    b = r.get("best")
    if b:
        print(f"\nBest sweep: ICC={b.get('icc'):.4f}  {b}")

diag = Path("research_log/severity_icc_diagnosis.json")
if diag.exists():
    r = json.loads(diag.read_text())
    print("\nDiagnosis tests:")
    for k, v in r.get("tests", {}).items():
        icc = v.get("icc")
        if icc is not None:
            print(f"  {k:28s} ICC={icc:.4f}  n={v.get('n_pairs')}")
PY

echo ""
echo "Reports under research_log/: severity_icc_{train,val,test}.json, severity_icc_diagnosis.json, icc_combine_sweep.json, slot_axis_icc_comparison.json"
