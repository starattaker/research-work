# 06 — Keypoint R-CNN training

**Machine:** Friend RTX ~12 GB (Pop!_OS)  
**Batch:** 4 (paper-equivalent on 12 GB)  
**Updated:** 2026-08-27

## Summary — test OKS (selected: **v6**)

| Model | v4 | v5 | **v6** | Paper |
|-------|---:|---:|-------:|------:|
| CEJ | 0.921 | — | **0.927** | 0.954 |
| Intersection | 0.822 | 0.859 | **0.894** | 0.912 |
| Apex | 0.853 | — | **0.871** | 0.815 |

**Decision:** **v6** wins all three vs v4/v5. Use for ICC pipeline.

## Experiment layout

| Ablation | Processed data | Weights |
|----------|----------------|---------|
| v4 | `data/processed_v4/` | `runs/keypoints/v4_*` |
| v5 | `data/processed_v5/` | `runs/keypoints/v5_*` (intersection only trained) |
| **v6** | `data/processed_v6/` | `runs/keypoints/v6_*` |

## v6 preprocessing rules

- **CEJ / apex:** v4 region growing → PCA long-axis slots (no x-sort pad)
- **Intersection:** bone-line endpoints → nearest mask; max 2 (left/right bone line)
- **Clamp:** keypoints clipped to image bounds before save/load

## v6 training (friend GPU, 2026-08-27)

```bash
bash scripts/run_v6_experiment.sh
```

| Model | test_oks | run_dir |
|-------|----------:|---------|
| cej | 0.927 | `runs/keypoints/v6_cej` |
| intersection | 0.894 | `runs/keypoints/v6_intersection` |
| apex | 0.871 | `runs/keypoints/v6_apex` |

## Notes

- Training uses **GT bboxes** from JSON; ICC uses **YOLO boxes** at inference.
- Registry: `research_log/experiments/paper_table.json`

## Next

End-to-end ICC: `scripts/run_severity_icc.py` — see [07_severity_icc.md](07_severity_icc.md).
