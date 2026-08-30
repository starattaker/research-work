#!/usr/bin/env bash
# ICC on train / val / test (paper defaults: roi + paper_x + both_sides).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
source venv/bin/activate 2>/dev/null || true

YOLO="${YOLO:-runs/detect/runs/detection/yolov8x_tooth/weights/best.pt}"
CEJ="${CEJ:-runs/keypoints/v6_cej/best.pt}"
INT="${INT:-runs/keypoints/v6_intersection/best.pt}"
APEX="${APEX:-runs/keypoints/v6_apex/best.pt}"
DATA="${DATA:-data/processed_v6}"

for split in train val test; do
  echo "========== ICC $split =========="
  python scripts/run_severity_icc.py \
    --yolo-weights "$YOLO" \
    --cej-weights "$CEJ" \
    --intersection-weights "$INT" \
    --apex-weights "$APEX" \
    --data-root "$DATA" \
    --split "$split" \
    --inference-mode roi \
    --combine-mode tensor \
    --gt-slot-convention pca \
    --severity-protocol both_sides \
    --out "research_log/severity_icc_${split}.json"
done

echo ""
echo "Summary:"
python - <<'PY'
import json
from pathlib import Path
for split in ("train", "val", "test"):
    p = Path(f"research_log/severity_icc_{split}.json")
    if not p.exists():
        print(f"  {split}: missing")
        continue
    r = json.loads(p.read_text())
    icc = r.get("icc")
    print(f"  {split}: ICC={icc:.4f}  n={r.get('n_pairs')}  target=0.801" if icc else f"  {split}: n/a")
PY
