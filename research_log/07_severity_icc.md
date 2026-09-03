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

\subsection{Keypoint model in ICC pipeline}
Severity is computed from three landmarks per root side: CEJ, bone intersection, and apex. **Intersection cannot be omitted** from ICC—it defines the crest position in the bone-loss ratio. What matters is the **checkpoint**:

| Checkpoint | Intersection test OKS | Use in ICC? |
|------------|----------------------:|-------------|
| **v6** | **0.894** | **Yes (production)** |
| v7 | 0.882 | **No** — worse OKS, overfit training |

All ICC scripts default to `runs/keypoints/v6_intersection/best.pt`.
