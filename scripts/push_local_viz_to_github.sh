#!/usr/bin/env bash
# Friend GPU: copy 5-image subset + weights + viz figures, then git push.
# Your PC:  git pull origin denpar-severity-replication
set -euo pipefail
cd "$(dirname "$0")/.."

STEMS=(1036 1140 201 797 944)
SPLIT=test
DATA_SRC=data/processed_v6
DATA_DST=data/local_viz_subset

echo "==> Copy 5-image label subset"
for stem in "${STEMS[@]}"; do
  for kpt in cej intersection apex; do
    mkdir -p "$DATA_DST/keypoints/$kpt/$SPLIT/images"
    mkdir -p "$DATA_DST/keypoints/$kpt/$SPLIT/annotations"
    cp "$DATA_SRC/keypoints/$kpt/$SPLIT/images/${stem}.jpg" "$DATA_DST/keypoints/$kpt/$SPLIT/images/"
    cp "$DATA_SRC/keypoints/$kpt/$SPLIT/annotations/${stem}.json" "$DATA_DST/keypoints/$kpt/$SPLIT/annotations/"
  done
  mkdir -p "$DATA_DST/yolo_detection/$SPLIT/images"
  if [[ -f "$DATA_SRC/yolo_detection/$SPLIT/images/${stem}.jpg" ]]; then
    cp "$DATA_SRC/yolo_detection/$SPLIT/images/${stem}.jpg" "$DATA_DST/yolo_detection/$SPLIT/images/"
  else
    cp "$DATA_SRC/keypoints/cej/$SPLIT/images/${stem}.jpg" "$DATA_DST/yolo_detection/$SPLIT/images/"
  fi
done

echo "==> Copy model weights (Git LFS)"
mkdir -p artifacts/local_viz
YOLO_W="runs/detect/runs/detection/yolov8x_tooth/weights/best.pt"
[[ -f "$YOLO_W" ]] || YOLO_W="runs/detection/yolov8x_tooth/weights/best.pt"
cp "$YOLO_W" artifacts/local_viz/yolo_best.pt
cp runs/keypoints/v6_cej/best.pt artifacts/local_viz/v6_cej_best.pt
cp runs/keypoints/v6_intersection/best.pt artifacts/local_viz/v6_intersection_best.pt
cp runs/keypoints/v6_apex/best.pt artifacts/local_viz/v6_apex_best.pt

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local 2>/dev/null || true
fi

echo "==> Stage & commit"
git add data/local_viz_subset artifacts/local_viz
git add research_log/figures/severity_pipeline_steps 2>/dev/null || true
git add .gitattributes

if git diff --cached --quiet; then
  echo "Nothing new to commit."
  exit 0
fi

git commit -m "Sync local viz bundle: 5 test images, weights, pipeline figures"
git push origin denpar-severity-replication

echo ""
echo "Done. On your PC:"
echo "  git pull origin denpar-severity-replication"
echo "  python scripts/visualize_severity_pipeline_steps.py \\"
echo "    --yolo-weights artifacts/local_viz/yolo_best.pt \\"
echo "    --cej-weights artifacts/local_viz/v6_cej_best.pt \\"
echo "    --intersection-weights artifacts/local_viz/v6_intersection_best.pt \\"
echo "    --apex-weights artifacts/local_viz/v6_apex_best.pt \\"
echo "    --data-root data/local_viz_subset --split test --n-images 5 --seed 42"
