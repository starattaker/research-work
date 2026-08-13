# 04 — Hyperparameters (paper replication)

All training scripts default to these values. **Do not change without noting in this log.**

## 1. Tooth detection — YOLOv8x

| Parameter | Paper / our script | Official repo note |
|-----------|-------------------|-------------------|
| Model | `yolov8x.pt` | Repo uses `yolov9e.pt` (ignored; paper specifies YOLOv8x) |
| Optimizer | Adam | Adam |
| Batch size | **1** (paper: 4; RTX 3050 4 GB VRAM — batch 2 OOM at epoch 4) | 4 |
| Initial LR (`lr0`) | 0.0001 | 0.0001 |
| LR schedule | Cosine (`cos_lr=True`) | Not set in repo script |
| Epochs | 200 | 200 |
| Image size | 640 | 640 |
| Early stopping patience | 25 | 25 |

**Script:** `scripts/train_detection.py`  
**Paper test targets:** mAP50 = 0.963, mAP50:95 = 0.907, precision = 0.892

### Augmentations (do not change)

Paper and official repo **do not override** YOLO augmentations — they use Ultralytics defaults via `model.train(...)`.

Our training run (`runs/.../args.yaml`) confirms:

| Setting | Value | Notes |
|---------|-------|-------|
| mosaic | 1.0 | **On** (standard YOLOv8) |
| close_mosaic | 10 | Off for last 10 epochs (Ultralytics default) |
| mixup / cutmix | 0.0 | Off (default) |
| fliplr | 0.5 | Default |
| hsv_h/s/v | 0.015 / 0.7 / 0.4 | Default |

`augment: false` in Ultralytics logs is **not** mosaic — it is a separate predict/val flag. Training mosaic remains enabled.

**Hardware-only overrides allowed:** `batch`, `workers`  
**Not allowed without paper justification:** mosaic, mixup, LR, optimizer, epochs, imgsz, etc.

## 2. Keypoint R-CNN (×3: CEJ, intersection, apex)

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 + FPN (`keypointrcnn_resnet50_fpn`) |
| Keypoints per tooth | 2 |
| Classes | 3 (background + single + double) |
| Optimizer | Adam, `lr=0.0001`, `weight_decay=1e-6` |
| LR scheduler | StepLR(`step_size=4`, `gamma=0.6`) |
| Train batch | **2** (paper: 8; **12 GB VRAM OOM at batch 4** on friend RTX ~12 GB, 2026-08-14) |
| Val/test batch | 2 |
| Epochs | 200 |
| Early stopping patience | 30 (validation loss; paper) |
| Augmentation | CLAHE `clip_limit=40`, `tile_grid_size=(8,8)`, p=1.0 |

**Scripts:** `scripts/train_keypoints.py`, `src/keypoint/train.py`

**Paper test targets (AP50:95, OKS):**

| Model | Target |
|-------|--------|
| CEJ | 0.954 |
| Intersection | 0.912 |
| Apex | 0.815 |

## 3. Inference / severity (post-training, not yet run)

| Parameter | Value |
|-----------|-------|
| NMS IoU (combine keypoints) | 0.6 |
| Severity formula | Min-max line + Eq. 1 (`src/severity/bone_loss.py`) |
| Evaluation metric | ICC vs ground truth |

**Paper test target:** ICC = 0.801
