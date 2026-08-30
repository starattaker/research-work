# 07 — Severity inference + ICC

**Status:** Honest test ICC **0.50–0.57** vs paper **0.801**  
**Updated:** 2026-08-31

## What ICC is measuring

GT severity is **not stored**. It is computed from `data/processed_v6/keypoints/{cej,intersection,apex}/{split}/annotations/*.json` with **PCA slot order** from preprocess.

Pred: YOLO box → Keypoint R-CNN ×3 (NMS 0.6) → pair 2 CEJ + 2 INT + 2 APEX → Eq. 1, clipped **[0, 100]**.

## Bugs found

1. **Wrong GT convention** — `paper_x` GT vs v6 PCA GT: ICC ≈ 0. Never use `paper_x` GT on processed_v6.
2. **Slot index ≠ anatomy** — model slot 0 is not GT PCA slot 0. Pairing by index vs nearest-CEJ changes ICC by ~0.05–0.15.
3. **CEJ pairing is not always better** — lr slot-index 0.56 vs CEJ-match 0.41 on the same run.
4. **Train/val ICC << test** (0.02 / 0.08 vs 0.50) — not a healthy pattern; treat test 0.57 as provisional until val-locked protocol matches.

## Numbers (friend GPU)

| Stage | Ours | Paper |
|-------|-----:|------:|
| YOLO mAP50 | 0.873 | 0.963 |
| CEJ OKS | 0.927 | 0.954 |
| Intersection OKS | 0.894 | 0.912 |
| Apex OKS | 0.871 | 0.815 |
| Severity ICC (honest) | **0.50–0.57** | **0.801** |
| Severity ICC (oracle 8-combo) | 0.79 | — |

## NMS

NMS @ IoU 0.6 is **within each Keypoint R-CNN**, on detection boxes — not heatmap peak picking across the three models.

## Command

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_icc_friend_full.sh
```

Output: `research_log/icc_final_report.json`
