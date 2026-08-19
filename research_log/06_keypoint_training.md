# 06 — Keypoint R-CNN training

**Machine:** Friend RTX ~12 GB (Pop!_OS)  
**Preprocessing on disk:** v1 (8 px margin) — `data/processed/`  
**Updated:** 2026-08-14

## CEJ (completed)

| Item | Value |
|------|-------|
| Command | `python -m src.keypoint.train --data-root data/processed/keypoints/cej --keypoint-type cej --output-dir runs/keypoints/cej --batch-size 4 --patience 30 --device cuda` |
| Epochs run | 35 (early stop) |
| Best epoch | **5** (lowest val loss) |
| Weights | `runs/keypoints/cej/best.pt` |
| **test_oks** | **0.820** |
| test_map_50 | 0.825 |
| Paper CEJ target | 0.954 |

## Intersection / apex

| Model | Status | Output dir |
|-------|--------|------------|
| Intersection | Pending | `runs/keypoints/intersection/` |
| Apex | Pending | `runs/keypoints/apex/` |

## Experiment layout (do not overwrite)

Use **separate folders** per preprocessing ablation:

| Ablation | Processed data | Keypoint weights |
|----------|----------------|------------------|
| v1 (8 px margin) | `data/processed/` (current) | `runs/keypoints/v1_{cej,intersection,apex}/` |
| v2 (strict bbox) | `data/processed_v2/` | `runs/keypoints/v2_*` |
| v3 (mask+grace) | `data/processed_v3/` | `runs/keypoints/v3_*` |

`best.pt` is saved automatically each time val loss improves; re-training the **same** `--output-dir` overwrites it. Use distinct `--output-dir` per experiment.

## Visual QA

```bash
python scripts/test_keypoint_detection.py --keypoint-type cej --n 10 --device cuda
python scripts/launch_tensorboard.py --logdir runs/keypoints
```

**Many red boxes?** Raw Keypoint R-CNN outputs ~30+ proposals per image. Use default NMS (now applied in viz script). Add `--show-raw` to see unfiltered output.

## Notes

- Training uses **GT bboxes from JSON**, not YOLO predictions.
- Keypoint viz red boxes = model detections before NMS; not necessarily training input.

## v2 keypoints (auto)

Recorded: 2026-08-19 23:24 UTC

| Model | test_oks | Paper | best epoch | run_dir |
|-------|----------:|------:|-----------:|---------|
| cej | 0.842615787088871 | 0.954 | 8 | `runs/keypoints/v2_cej` |
| intersection | 0.8145487064123154 | 0.912 | 6 | `runs/keypoints/v2_intersection` |
| apex | 0.7811488563381136 | 0.815 | 5 | `runs/keypoints/v2_apex` |
