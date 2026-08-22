# 06 — Keypoint R-CNN training

**Machine:** Friend RTX ~12 GB (Pop!_OS)  
**Batch:** 4 (paper-equivalent on 12 GB)  
**Updated:** 2026-08-22

## Summary — test OKS by preprocessing ablation

| Model | v1 | v2 | v3 | **v4** | Paper |
|-------|---:|---:|---:|-------:|------:|
| CEJ | 0.820* | 0.843 | 0.911 | **0.921** | 0.954 |
| Intersection | — | 0.815 | 0.817 | **0.822** | 0.912 |
| Apex | — | 0.781 | 0.836 | **0.853** | 0.815 |

\*v1 CEJ only.

**Decision:** **v4 (region growing)** wins all three vs v2/v3. **v5** (endpoint intersections) training next.

## Experiment layout

| Ablation | Processed data | Weights |
|----------|----------------|---------|
| v1 | `data/processed/` | `runs/keypoints/cej/` |
| v2 | `data/processed_v2/` | `runs/keypoints/v2_*` |
| v3 | `data/processed_v3/` | `runs/keypoints/v3_*` |
| v4 | `data/processed_v4/` | `runs/keypoints/v4_*` |
| **v5** | `data/processed_v5/` | `runs/keypoints/v5_*` |

## v4 keypoints

Recorded: 2026-08-20 07:03 UTC

| Model | test_oks | best epoch | run_dir |
|-------|----------:|-----------:|---------|
| cej | 0.921 | 5 | `runs/keypoints/v4_cej` |
| intersection | 0.822 | 5 | `runs/keypoints/v4_intersection` |
| apex | 0.853 | 6 | `runs/keypoints/v4_apex` |

## v5 keypoints

**Status:** labels ready; training not started.

```bash
bash scripts/run_experiment_v5.sh
```

## Command (per model)

```bash
python -m src.keypoint.train \
  --data-root data/processed_v5/keypoints/cej \
  --keypoint-type cej \
  --output-dir runs/keypoints/v5_cej \
  --batch-size 4 --patience 30 --device cuda
```

## Notes

- Training uses **GT bboxes from JSON**, not YOLO predictions.
- Keypoint test OKS is on GT boxes; ICC requires **YOLO boxes at inference**.
- Registry: `research_log/experiments/paper_table.json`
