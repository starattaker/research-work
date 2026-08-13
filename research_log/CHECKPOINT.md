# Checkpoint — YOLO training started (2026-08-13)

**Status:** YOLOv8x tooth detection **in progress**. Keypoint R-CNN **not started**.

## Completed

- [x] venv + dependencies (CUDA)
- [x] DenPAR downloaded and preprocessed → `data/processed/`
- [x] Validation passed
- [x] Batch sizes reduced for 16 GB VRAM (YOLO=2, Keypoint train=4)
- [x] Research log + `make-a-checkpoint` skill

## In progress

- [ ] **YOLOv8x tooth detection** — `runs/detection/yolov8x_tooth/`
  - Log: terminal / Ultralytics output
  - Doc: [05_detection_training.md](05_detection_training.md)

## Not started

- [ ] Keypoint R-CNN (CEJ → intersection → apex)
- [ ] Full inference pipeline
- [ ] Severity ICC evaluation (target 0.801)

## After YOLO finishes

1. Record test metrics in `05_detection_training.md` vs paper targets
2. Run **Make a checkpoint** (or ask agent to) to refresh this file
3. When approved, train keypoints one model at a time:

```powershell
.\venv\Scripts\python.exe -m src.keypoint.train --data-root data/processed/keypoints/cej --keypoint-type cej --output-dir runs/keypoints/cej
```

## Monitor YOLO

```powershell
# Training already running; check runs/detection/yolov8x_tooth/results.csv
```
