# 05 — YOLOv8x tooth detection training

**Started:** 2026-08-13  
**Status:** In progress

## Command

```powershell
.\venv\Scripts\python.exe scripts\train_detection.py
```

## Hyperparameters (this run)

| Parameter | Paper | This run |
|-----------|-------|----------|
| Model | YOLOv8x | YOLOv8x |
| Optimizer | Adam | Adam |
| Batch size | 4 | **1** (OOM fix) |
| lr0 | 0.0001 | 0.0001 |
| LR schedule | Cosine | Cosine |
| Epochs | 200 | 200 |
| imgsz | 640 | 640 |
| Patience | 25 | 25 |

## Data

- `data/processed/yolo_detection/data.yaml`
- Classes: `single` (0), `double` (1)

## Output

- `runs/detection/yolov8x_tooth/`
- Best weights: `runs/detection/yolov8x_tooth/weights/best.pt`

## Paper targets (test set)

| Metric | Target |
|--------|--------|
| mAP50 | 0.963 |
| mAP50:95 | 0.907 |
| Precision | 0.892 |

## Results (partial — crashed epoch 4)

| Epoch | mAP50 | mAP50:95 |
|-------|-------|----------|
| 1 | 0.824 | 0.721 |
| 2 | 0.811 | 0.721 |
| 3 | 0.835 | 0.735 |

Crash: `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` (batch=2, ~2.8 GB VRAM).  
Resuming from `runs/detect/runs/detection/yolov8x_tooth/weights/last.pt` with **batch=1**.

## Results (final)

*(Fill in when training completes)*
