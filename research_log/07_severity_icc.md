# 07 — Severity inference + ICC

**Status:** Not started  
**Target:** ICC = **0.801** (paper)  
**Selected preprocessing:** **v4** (region growing)  
**Updated:** 2026-08-20

## Pipeline (paper order)

```
Panoramic X-ray
  → YOLOv8x tooth detection (best.pt)
  → Per-tooth Keypoint R-CNN ×3 (v4_cej, v4_intersection, v4_apex)
  → NMS IoU 0.6 (combine duplicate tooth/keypoint proposals)
  → Min-max line + Eq. 1 severity % (src/severity/bone_loss.py)
  → ICC vs expert severity labels (DenPAR)
```

## Inputs required

| Asset | Path (friend GPU) |
|-------|-------------------|
| YOLO weights | `runs/detect/train/weights/best.pt` |
| v4 CEJ | `runs/keypoints/v4_cej/best.pt` |
| v4 intersection | `runs/keypoints/v4_intersection/best.pt` |
| v4 apex | `runs/keypoints/v4_apex/best.pt` |
| Test images | `data/processed_v4/yolo_detection/test/images/` |
| GT severity | DenPAR severity labels (see `02_dataset.md`) |

## Parameters (replication — do not change)

| Parameter | Value |
|-----------|-------|
| Keypoint NMS IoU | 0.6 |
| Score threshold | 0.5 (viz / inference default) |
| Severity formula | `compute_bone_loss_severity()` in `src/severity/bone_loss.py` |

## What exists today

| Component | Status |
|-----------|--------|
| Severity math | `src/severity/bone_loss.py` |
| Keypoint inference + NMS | `src/keypoint/inference_utils.py`, `scripts/test_keypoint_detection.py` |
| YOLO inference | `scripts/test_yolo_detection.py` |
| **End-to-end script** | **Not implemented** — needs `scripts/run_severity_icc.py` (or equivalent) |

## Expected blockers for ICC

1. **YOLO box error** — keypoint OKS was measured on GT boxes; real ICC uses predicted boxes.
2. **Tooth matching** — assign predicted teeth to GT for ICC pairing.
3. **Missing keypoints** — severity `None` when CEJ/intersection/apex invisible.
4. **Double-root teeth** — weaker YOLO class may hurt multi-apex cases.

## Next command (when script exists)

```bash
cd ~/faraz/Test_work/research-work
export PYTHONPATH=.
python scripts/run_severity_icc.py \
  --yolo-weights runs/detect/train/weights/best.pt \
  --cej-weights runs/keypoints/v4_cej/best.pt \
  --intersection-weights runs/keypoints/v4_intersection/best.pt \
  --apex-weights runs/keypoints/v4_apex/best.pt \
  --data-root data/processed_v4 \
  --split test
```

## Paper comparison table (keypoints done, ICC pending)

| Stage | Metric | Best (v4) | Paper |
|-------|--------|----------:|------:|
| YOLO | mAP50 | 0.873 | 0.963 |
| CEJ | test OKS | 0.921 | 0.954 |
| Intersection | test OKS | 0.822 | 0.912 |
| Apex | test OKS | 0.853 | 0.815 |
| Severity | ICC | — | 0.801 |
