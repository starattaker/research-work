#!/usr/bin/env bash
# Friend GPU: sync repo + ICC pipeline (v6 PCA GT + tensor combine + both sides).
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/sync_friend_repo.sh

export PYTHONPATH=.
unset RAW_ROOT 2>/dev/null || true
source venv/bin/activate

YOLO="${YOLO:-runs/detect/runs/detection/yolov8x_tooth/weights/best.pt}"
CEJ="${CEJ:-runs/keypoints/v6_cej/best.pt}"
INT="${INT:-runs/keypoints/v6_intersection/best.pt}"
APEX="${APEX:-runs/keypoints/v6_apex/best.pt}"
DATA="${DATA:-data/processed_v6}"
DEVICE="${DEVICE:-cuda}"

echo "========== [1/4] GT sanity (PCA vs paper_x on v6 labels) =========="
python scripts/icc_gt_sanity.py --data-root "$DATA" --split test \
  --out research_log/icc_gt_sanity_test.json

echo ""
echo "========== [2/4] ICC diagnosis =========="
python scripts/diagnose_severity_icc.py \
  --data-root "$DATA" --split test --device "$DEVICE" \
  --yolo-weights "$YOLO" --cej-weights "$CEJ" \
  --intersection-weights "$INT" --apex-weights "$APEX" \
  --inference-mode roi \
  --out research_log/severity_icc_diagnosis.json

echo ""
echo "========== [3/4] ICC train / val / test (pca GT + tensor + both_sides) =========="
for split in train val test; do
  python scripts/run_severity_icc.py \
    --data-root "$DATA" --split "$split" --device "$DEVICE" \
    --yolo-weights "$YOLO" --cej-weights "$CEJ" \
    --intersection-weights "$INT" --apex-weights "$APEX" \
    --inference-mode roi --combine-mode tensor \
    --gt-slot-convention pca --severity-protocol both_sides \
    --out "research_log/severity_icc_${split}.json"
done

echo ""
echo "========== [4/4] Mode sweep (test) =========="
python scripts/sweep_icc_combine.py \
  --data-root "$DATA" --split test --device "$DEVICE" \
  --yolo-weights "$YOLO" --cej-weights "$CEJ" \
  --intersection-weights "$INT" --apex-weights "$APEX" \
  --out research_log/icc_combine_sweep.json

python - <<'PY'
import json
from pathlib import Path
print("\n========== SUMMARY ==========")
print("ICC (pca GT, tensor pred, both_sides, roi):")
for split in ("train", "val", "test"):
    p = Path(f"research_log/severity_icc_{split}.json")
    if p.exists():
        r = json.loads(p.read_text())
        icc = r.get("icc")
        print(f"  {split:5s}  ICC={icc:.4f}  n={r.get('n_pairs')}  target test=0.801" if icc else f"  {split}: n/a")
sp = Path("research_log/icc_combine_sweep.json")
if sp.exists():
    b = json.loads(sp.read_text()).get("best")
    if b:
        print(f"\nBest sweep: ICC={b['icc']:.4f}  n={b['n_pairs']}  {b['combine_mode']} gt={b['gt_slot_convention']}")
san = Path("research_log/icc_gt_sanity_test.json")
if san.exists():
    s = json.loads(san.read_text())
    print(f"\nGT sanity PCA vs paper_x ICC={s.get('icc_pca_slot0_vs_paper_x')} (≈0 confirms v6 uses PCA slots)")
PY
