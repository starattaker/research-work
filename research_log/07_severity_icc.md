# 07 — Severity inference + ICC

**Status:** Test ICC **~0.69–0.73** vs paper **0.801**  
**Updated:** 2026-09-02

## Pipeline

GT severity from `processed_v6` keypoint JSON (**PCA slots**). Pred: YOLO → 3× Keypoint R-CNN (NMS 0.6) → combine → Eq. 1, clipped **[0, 100]**, geom filter.

## ICC results

| Run | Test ICC | Val ICC | Config |
|-----|--------:|--------:|--------|
| Initial fix (`861dfca`) | 0.7005 / **0.7283** | 0.7048 | hungarian / tensor + match_by_slot |
| Parameter grid (`146aa80`) | **0.6904** / **0.7332** | **0.7338** | val-locked hungarian apex28 / test-best tensor apex8 |

| Metric | Ours (v6) | Paper |
|--------|----------:|------:|
| YOLO mAP50 | 0.873 | 0.963 |
| CEJ OKS | 0.927 | 0.954 |
| Intersection OKS | 0.894 | 0.912 |
| Apex OKS | 0.871 | 0.815 |
| Severity ICC (honest) | **~0.73** | **0.801** |
| Oracle 8-combo | 0.79 | — |

**Production recommendation:** `tensor` + `match_by_slot` + `apex_merge_px=8` + `pca` GT (do not use val-locked hungarian apex28 on test).

## Parameter grid notes (126 configs)

- Top val: hungarian + match_by_slot, apex 8–32px (ICC ~0.733, flat)
- mask_pca + both_sides ~0.716 (worse)
- Val↑ test↓ when locking hungarian apex28 → mild val overfit on protocol

## Preprocess sweep (for v7)

| Knob | v6 | Sweep recommendation |
|------|---:|---------------------|
| max_grace_px | 8 | **12** |
| bbox_outlier_margin_px | — | **7.6** |
| 100% point assign | — | 22px (0.03% wrong) |

Reports: `research_log/icc_parameter_sweep.json`, `research_log/point_assignment_report.json`

## v7 (optional ICC)

v7 weights complete; intersection OKS regressed. One-shot eval:

```bash
python scripts/run_icc_parameter_sweep.py --cej-weights runs/keypoints/v7_cej/best.pt \
  --intersection-weights runs/keypoints/v7_intersection/best.pt \
  --apex-weights runs/keypoints/v7_apex/best.pt --out research_log/icc_v7_report.json
```
