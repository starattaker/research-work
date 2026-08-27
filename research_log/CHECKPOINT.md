# Checkpoint — v6 selected, ICC script ready (2026-08-27)

**Status:** v6 keypoints **best** on all three heads. Run end-to-end ICC next.

## Completed

- [x] YOLO — test mAP50 **0.873** (paper 0.963)
- [x] Keypoint ablations v2–v6; **v6 wins all three**
- [x] v5 intersection **0.859** · v6 intersection **0.894** (paper 0.912)
- [x] **`scripts/run_severity_icc.py`** — YOLO + 3× Keypoint R-CNN + Eq. 1

## v6 test OKS (friend GPU, 2026-08-27)

| Model | v4 | v6 | Paper |
|-------|---:|---:|------:|
| CEJ | 0.921 | **0.927** | 0.954 |
| Intersection | 0.822 | **0.894** | 0.912 |
| Apex | 0.853 | **0.871** | 0.815 |

**Selected stack:** `data/processed_v6/` + `runs/keypoints/v6_{cej,intersection,apex}/best.pt`

## In progress

- [ ] **End-to-end ICC** on DenPAR test (target **0.801**)

## Not started

- [ ] Team 214-image external validation
- [ ] Update progress paper (LaTeX) with v6 table

## Resume — friend GPU

```bash
cd ~/faraz/Test_work/research-work
git pull origin denpar-severity-replication
export PYTHONPATH=.
python scripts/run_severity_icc.py \
  --yolo-weights runs/detect/runs/detection/yolov8x_tooth/weights/best.pt \
  --cej-weights runs/keypoints/v6_cej/best.pt \
  --intersection-weights runs/keypoints/v6_intersection/best.pt \
  --apex-weights runs/keypoints/v6_apex/best.pt \
  --data-root data/processed_v6 \
  --split test
```

Results → `research_log/severity_icc_end_to_end.json`
