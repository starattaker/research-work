# 06 — Keypoint R-CNN training

**Machine:** Friend RTX ~12 GB (Pop!_OS)  
**Batch:** 4 (paper-equivalent on 12 GB)  
**Updated:** 2026-09-03

## Summary — test OKS

| Model | v4 | v5 | **v6** | v7 | Paper |
|-------|---:|---:|-------:|---:|------:|
| CEJ | 0.921 | — | **0.927** | 0.928 | 0.954 |
| Intersection | 0.822 | 0.859 | **0.894** | 0.882 | 0.912 |
| Apex | 0.853 | — | **0.871** | 0.881 | 0.815 |

**Decision:** **v6** for ICC pipeline. v7 documents extended grace sweep; intersection regressed.

## Experiment layout

| Ablation | Processed data | Weights |
|----------|----------------|---------|
| v4 | `data/processed_v4/` | `runs/keypoints/v4_*` |
| v5 | `data/processed_v5/` | `runs/keypoints/v5_intersection` |
| **v6** | `data/processed_v6/` | `runs/keypoints/v6_*` |
| v7 | `data/processed_v7/` | `runs/keypoints/v7_*` |

## v6 (selected)

- CEJ/apex: region grow 8px → **PCA-axis slots**
- Intersection: bone-line endpoints → nearest mask (L/R)
- Weights: `runs/keypoints/v6_{cej,intersection,apex}/best.pt`

## v7 (2026-09-02/03)

Preprocess: `max_grace_px=12`, `bbox_outlier_margin_px=7.6` from point-assignment sweep.

```bash
bash scripts/run_train_v7_from_sweep_friend.sh
```

All three heads complete. Overfit pattern same as v6 runs (best ~epoch 4–5).

## Notes

- Training uses **GT bboxes**; ICC uses **YOLO** at inference.
- Registry: `research_log/experiments/paper_table.json`

## v6 keypoints (auto)

Recorded: 2026-09-12 18:44 UTC

| Model | test_oks | Paper | best epoch | run_dir |
|-------|----------:|------:|-----------:|---------|
| cej | 0.9210635432600975 | 0.954 | 5 | `runs/keypoints/v4_cej` |
| intersection | 0.817021611481905 | 0.912 | 5 | `runs/keypoints/v3_intersection` |
| apex | 0.8530984434485436 | 0.815 | 6 | `runs/keypoints/v4_apex` |
