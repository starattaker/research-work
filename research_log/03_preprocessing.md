# 03 — Preprocessing

**Script:** `src/preprocess/prepare_dataset.py`  
**Validation:** `scripts/validate_preprocessed.py`  
**Output:** `data/processed/`

## Outputs

### YOLO detection (`yolo_detection/`)

- Classes: `0 = single`, `1 = double` (from apex count in bbox)
- Splits: `train/`, `val/`, `test/` with `images/` + `labels/`
- Config: `data/processed/yolo_detection/data.yaml`

### Keypoint R-CNN (`keypoints/{cej,intersection,apex}/`)

Per-image JSON in `annotations/` with:
- `bboxes`: Pascal VOC `[x1,y1,x2,y2]`
- `labels`: `1` = single, `2` = double (mapped to model class indices 1/2)
- `keypoints`: 2 points per tooth, `[x, y, visibility]` (COCO-style), sorted by x

## Validated counts (2026-08-12)

| Split | Images | YOLO labels | Teeth | CEJ / Int / Apex JSON |
|-------|--------|-------------|-------|------------------------|
| train | 650 | 650 | 2,883 | 650 each |
| val | 150 | 150 | 655 | 150 each |
| test | 200 | 200 | 864 | 200 each |

All image ↔ annotation pairs match (no missing bone annotations: `missing_bone = 0`).

## Logic notes

1. **CEJ / apex → teeth:** assigned by bbox proximity (`margin = 8 px`).
2. **Intersection points:** line–contour intersection of bone level line with tooth mask contour; nearest contour point fallback.
3. **Invalid keypoints:** `(0, 0, 0)` when point missing or outside tooth.

## Re-run

```powershell
.\venv\Scripts\python.exe -m src.preprocess.prepare_dataset
.\venv\Scripts\python.exe scripts\validate_preprocessed.py
```
