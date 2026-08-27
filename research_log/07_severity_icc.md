# 07 — Severity inference + ICC

**Status:** Script ready — run on friend GPU  
**Target:** ICC = **0.801** (paper)  
**Selected preprocessing:** **v6** (PCA-axis CEJ/apex + L/R endpoint intersections)  
**Updated:** 2026-08-27

## Pipeline (paper order)

```
Panoramic X-ray
  → YOLOv8x tooth detection (best.pt)
  → Match YOLO box to GT tooth (IoU ≥ 0.5)
  → YOLO box as fixed ROI proposal → Keypoint R-CNN ×3 (no RPN / no R-CNN tooth detector)
  → NMS IoU 0.6 + score ≥ 0.5 (per model, paper)
  → Combine CEJ + intersection + apex (v6 aligned slots) → Eq. 1 severity %
  → ICC vs GT severity from processed label JSON
```

Diagnostic: `--no-require-yolo` uses **GT box** as the ROI proposal when YOLO misses (isolates keypoint error).

## Command

```bash
cd ~/faraz/Test_work/research-work
git pull origin denpar-severity-replication
export PYTHONPATH=.
python scripts/run_severity_icc.py \
  --yolo-weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt \
  --cej-weights runs/keypoints/v6_cej/best.pt \
  --intersection-weights runs/keypoints/v6_intersection/best.pt \
  --apex-weights runs/keypoints/v6_apex/best.pt \
  --data-root data/processed_v6 \
  --split test
```

**Output:** `research_log/severity_icc_end_to_end.json`

## Parameters (replication)

| Parameter | Value |
|-----------|-------|
| Keypoint NMS IoU | 0.6 |
| Score threshold | 0.5 |
| GT↔YOLO match IoU | 0.5 |
| Proposal↔ROI det IoU | 0.3 |
| Severity formula | `compute_bone_loss_severity()` |

## GT vs pred severity

| | Source |
|--|--------|
| **GT severity** | Processed v6 label JSON on **GT boxes** (CEJ + intersection + apex) |
| **Pred severity** | **YOLO boxes** + model keypoints + same Eq. 1 |

ICC therefore includes YOLO localization error (intended end-to-end metric).

## Paper comparison (keypoints done, ICC pending)

| Stage | Metric | Best (v6) | Paper |
|-------|--------|----------:|------:|
| YOLO | mAP50 | 0.873 | 0.963 |
| CEJ | test OKS | 0.927 | 0.954 |
| Intersection | test OKS | 0.894 | 0.912 |
| Apex | test OKS | 0.871 | 0.815 |
| Severity | ICC | — | 0.801 |
