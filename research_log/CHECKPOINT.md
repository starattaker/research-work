# Checkpoint — v5 preprocess ready (2026-08-22)

**Status:** v4 training complete. **v5 labels built** — train on friend GPU next.

## Completed

- [x] YOLO — test mAP50 **0.873** (paper 0.963)
- [x] Keypoint R-CNN v2/v3/v4 — **v4 best** (CEJ 0.921, int 0.822, apex **0.853**)
- [x] Intersection QA + ray-bug analysis ([intersection_logic_analysis.md](intersection_logic_analysis.md))
- [x] **v5 preprocess** — endpoint intersections → `data/processed_v5/` (rebuild on GPU with script)
- [x] QA scripts + `scripts/run_experiment_v5.sh`

## In progress

- [ ] **v5 keypoint training** on friend GPU

## Not started

- [ ] v6 (midplane CEJ + region grow)
- [ ] End-to-end inference + ICC (target **0.801**)

## v4 metrics (auto)

- CEJ 0.921 · intersection 0.822 · apex **0.853**
- Registry: `research_log/experiments/paper_table.json`

## Resume — friend GPU

```bash
cd ~/faraz/Test_work/research-work
git pull origin denpar-severity-replication
bash scripts/run_experiment_v5.sh
# or: SKIP_PREPROCESS=1 bash scripts/run_experiment_v5.sh
```

Weights → `runs/keypoints/v5_{cej,intersection,apex}/`
