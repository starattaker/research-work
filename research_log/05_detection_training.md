# 05 — YOLOv8x tooth detection training

**Completed:** 2026-08-13 (friend's RTX 5070 — Pop!_OS)  
**Status:** Done — early stopped epoch 67, best epoch 42

## Command (Linux)

```bash
python scripts/train_detection.py --batch 4 --workers 8
```

## Hyperparameters

| Parameter | Paper | This run |
|-----------|-------|----------|
| Model | YOLOv8x | YOLOv8x |
| Optimizer | Adam | Adam |
| Batch size | 4 | 4 |
| lr0 | 0.0001 | 0.0001 |
| cos_lr | Yes | Yes |
| Epochs (max) | 200 | 200 |
| Early stop patience | 25 | 25 |
| imgsz | 640 | 640 |
| **Epochs run** | — | **67** (best @ **42**) |

## Validation (best.pt, 150 images)

| Class | P | R | mAP50 | mAP50-95 |
|-------|---|----|-------|----------|
| all | 0.814 | 0.841 | 0.893 | 0.805 |
| single | 0.915 | 0.832 | 0.933 | 0.806 |
| double | 0.712 | 0.851 | 0.853 | 0.805 |

## Test (best.pt, 200 images) — compare to paper here

| Class | P | R | mAP50 | mAP50-95 |
|-------|---|----|-------|----------|
| all | 0.850 | 0.844 | **0.873** | **0.794** |
| single | 0.906 | 0.854 | 0.919 | 0.798 |
| double | 0.793 | 0.834 | 0.827 | 0.790 |

## Paper test targets

| Metric | Target | Ours | Gap |
|--------|--------|------|-----|
| mAP50 | 0.963 | 0.873 | −0.09 |
| mAP50-95 | 0.907 | 0.794 | −0.11 |
| Precision | 0.892 | 0.850 | −0.04 |

## Notes

- Early stopping is **paper-correct** (patience=25); no val gain after epoch 42.
- Main weakness: **double-root** class (230 test instances vs 634 single).
- mAP50-95 gap → boxes detected but localization not as tight as paper.
- Weights: `runs/detect/runs/detection/yolov8x_tooth/weights/best.pt`

## Next step

Proceed to Keypoint R-CNN (CEJ → intersection → apex) using `best.pt` for inference later.
