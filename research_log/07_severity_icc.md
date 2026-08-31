# 07 — Severity inference + ICC

**Status:** Test ICC **~0.70–0.73** vs paper **0.801**  
**Updated:** 2026-08-31

## Pipeline

GT severity from `processed_v6` keypoint JSON (**PCA slots**). Pred: YOLO → 3× Keypoint R-CNN (NMS 0.6) → combine → Eq. 1, clipped **[0, 100]**, geom filter (INT between CEJ and apex).

## Results (friend GPU, `861dfca`)

| Metric | Ours | Paper |
|--------|-----:|------:|
| YOLO mAP50 | 0.873 | 0.963 |
| CEJ OKS | 0.927 | 0.954 |
| Intersection OKS | 0.894 | 0.912 |
| Apex OKS | 0.871 | 0.815 |
| ICC test (val-locked) | **0.7005** | 0.801 |
| ICC test (best config) | **0.7283** | 0.801 |
| ICC val | 0.7048 | — |
| ICC train | 0.8279 | — |
| Oracle 8-combo | 0.79 | — |

**Best test config:** `tensor` + `match_by_slot` + `pca` GT.  
**Val-locked winner:** `hungarian` + `match_by_slot`.

## What fixed ICC (0.05 → 0.73)

1. GT = **pca** slots (not `paper_x`)
2. Pair **match_by_slot** (not CEJ-nearest)
3. Clip severity 0–100; reject invalid geometry

## Commands

```bash
# Full val-locked sweep
bash scripts/run_improvement_friend.sh
bash scripts/run_icc_friend_full.sh

# Parameter grid
bash scripts/run_icc_optimize_friend.sh
```

Reports: `research_log/icc_final_report.json`, `research_log/icc_parameter_sweep.json`
