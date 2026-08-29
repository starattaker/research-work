#!/usr/bin/env bash
# Run on friend GPU after training. Packs weights + 5 QA images for local Windows viz.
set -euo pipefail
cd "$(dirname "$0")/.."

# Same 5 images as visualize_severity_pipeline_steps.py --seed 42
STEMS=(1036 1140 201 797 944)
SPLIT=test
BUNDLE=local_viz_bundle
ARCHIVE=local_viz_bundle.tar.gz

rm -rf "$BUNDLE" "$ARCHIVE"
mkdir -p "$BUNDLE/runs/detect" "$BUNDLE/runs/keypoints" "$BUNDLE/data/processed_v6"

YOLO_W="runs/detect/runs/detection/yolov8x_tooth/weights/best.pt"
if [[ ! -f "$YOLO_W" ]]; then
  YOLO_W="runs/detection/yolov8x_tooth/weights/best.pt"
fi
cp "$YOLO_W" "$BUNDLE/runs/detect/best.pt"
cp runs/keypoints/v6_cej/best.pt "$BUNDLE/runs/keypoints/v6_cej_best.pt"
cp runs/keypoints/v6_intersection/best.pt "$BUNDLE/runs/keypoints/v6_intersection_best.pt"
cp runs/keypoints/v6_apex/best.pt "$BUNDLE/runs/keypoints/v6_apex_best.pt"

DATA=data/processed_v6
for stem in "${STEMS[@]}"; do
  for kpt in cej intersection apex; do
    mkdir -p "$BUNDLE/$DATA/keypoints/$kpt/$SPLIT/images"
    mkdir -p "$BUNDLE/$DATA/keypoints/$kpt/$SPLIT/annotations"
    cp "$DATA/keypoints/$kpt/$SPLIT/images/${stem}.jpg" "$BUNDLE/$DATA/keypoints/$kpt/$SPLIT/images/"
    cp "$DATA/keypoints/$kpt/$SPLIT/annotations/${stem}.json" "$BUNDLE/$DATA/keypoints/$kpt/$SPLIT/annotations/"
  done
  mkdir -p "$BUNDLE/$DATA/yolo_detection/$SPLIT/images"
  cp "$DATA/yolo_detection/$SPLIT/images/${stem}.jpg" "$BUNDLE/$DATA/yolo_detection/$SPLIT/images/" 2>/dev/null || \
    cp "$DATA/keypoints/cej/$SPLIT/images/${stem}.jpg" "$BUNDLE/$DATA/yolo_detection/$SPLIT/images/"
done

if [[ -d research_log/figures/severity_pipeline_steps ]]; then
  cp -r research_log/figures/severity_pipeline_steps "$BUNDLE/gpu_figures"
fi

cat > "$BUNDLE/manifest.json" <<'EOF'
{
  "stems": ["1036", "1140", "201", "797", "944"],
  "split": "test",
  "yolo_weights": "runs/detect/best.pt",
  "cej_weights": "runs/keypoints/v6_cej_best.pt",
  "intersection_weights": "runs/keypoints/v6_intersection_best.pt",
  "apex_weights": "runs/keypoints/v6_apex_best.pt",
  "data_root": "data/processed_v6"
}
EOF

tar czf "$ARCHIVE" "$BUNDLE"
echo "Created $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo "Copy to your PC, e.g.:"
echo "  scp $ARCHIVE YOUR_PC:~/Oralvis_Seekright/"
