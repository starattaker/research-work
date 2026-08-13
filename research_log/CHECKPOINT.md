# Checkpoint — CEJ keypoint done; ablation in progress (2026-08-14)

**Status:** CEJ Keypoint R-CNN **eval complete**. Intersection + apex **pending**. Preprocessing ablation v1→v2→v3 planned.

## Completed

- [x] YOLOv8x detection — test mAP50 **0.873** ([05_detection_training.md](05_detection_training.md))
- [x] CEJ Keypoint R-CNN trained + eval — test_oks **0.820** ([06_keypoint_training.md](06_keypoint_training.md))
- [x] Eval crash fix (`--eval-only`, metrics serialization)
- [x] TensorBoard + viz scripts (`test_yolo_detection.py`, `test_keypoint_detection.py`)
- [x] v3 preprocessing implemented (`--strategy v3`)

## In progress (friend machine)

- [ ] Intersection + apex keypoint training on **v1** data
- [ ] Log metrics to ablation table

## Not started

- [ ] Re-preprocess v2 → `data/processed_v2/` + train `runs/keypoints/v2_*`
- [ ] Re-preprocess v3 → `data/processed_v3/` + train `runs/keypoints/v3_*`
- [ ] Full inference (YOLO + 3 keypoint models + NMS 0.6)
- [ ] Severity ICC (target 0.801)

## Experiment tracking (friend machine)

| What | Auto-saved? | Overwrite risk |
|------|-------------|----------------|
| `best.pt` | Yes, on val improvement | **Yes** if same `--output-dir` |
| `history.json`, `metrics.json` | Yes, each run | Overwritten same dir |
| `tensorboard/` | Yes | Overwritten same dir |
| Processed data | On `prepare_dataset` | Overwritten if same `--output-root` |

**Rule:** one folder per experiment — never reuse `runs/keypoints/cej` for v2/v3.

## Resume commands (Linux)

```bash
cd ~/faraz/Test_work/research-work
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Finish v1 keypoints
python -m src.keypoint.train --data-root data/processed/keypoints/intersection \
  --keypoint-type intersection --output-dir runs/keypoints/intersection \
  --batch-size 4 --patience 30 --device cuda

python -m src.keypoint.train --data-root data/processed/keypoints/apex \
  --keypoint-type apex --output-dir runs/keypoints/apex \
  --batch-size 4 --patience 30 --device cuda

# Preprocessing comparison (no train)
python scripts/compare_preprocessing.py

# v2 data (separate folder)
python -m src.preprocess.prepare_dataset --strategy v2 --output-root data/processed_v2

# v3 data
python -m src.preprocess.prepare_dataset --strategy v3 --output-root data/processed_v3 --grace-px 4
```

## Key metrics snapshot

| Stage | Metric | Ours | Paper |
|-------|--------|------|-------|
| YOLO test | mAP50 | 0.873 | 0.963 |
| CEJ test | OKS | 0.820 | 0.954 |
