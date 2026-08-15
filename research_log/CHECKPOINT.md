# Checkpoint — v1 keypoints done; v2/v3 ablation next (2026-08-14)

**Status:** v1 keypoint training **complete** on friend GPU. Collect metrics → v2 preprocess → v2 train.

## Completed (friend machine)

- [x] YOLO — test mAP50 **0.873**
- [x] Keypoint R-CNN v1 data — CEJ + intersection + apex (confirm with collect script)

## One command after training (friend machine)

```bash
cd ~/faraz/Test_work/research-work
python scripts/collect_training_results.py
# optional: commit summary (small files only)
python scripts/collect_training_results.py --push-log
git push
```

Outputs: `research_log/metrics_snapshot.txt` — paste into chat or open on Windows after pull.

**Do not git-push `best.pt`** (large). Only `metrics.json` summaries if needed.

## Folder strategy — **same repo, NOT a new clone**

| Experiment | Processed data | Train output |
|------------|----------------|--------------|
| v1 (done) | `data/processed/` | `runs/keypoints/cej`, `intersection`, `apex` |
| v2 | `data/processed_v2/` | `runs/keypoints/v2_cej`, `v2_intersection`, `v2_apex` |
| v3 | `data/processed_v3/` | `runs/keypoints/v3_*` |

Same git clone; different `--output-root` and `--output-dir`.

## v2 on friend machine (next)

```bash
export PYTHONPATH=.

# 1. Preprocess v2 (strict bbox)
python -m src.preprocess.prepare_dataset --strategy v2 --output-root data/processed_v2

# 2. Train all three (separate dirs)
for KPT in cej intersection apex; do
  python -m src.keypoint.train \
    --data-root data/processed_v2/keypoints/$KPT \
    --keypoint-type $KPT \
    --output-dir runs/keypoints/v2_$KPT \
    --batch-size 4 --patience 30 --device cuda
done

python scripts/collect_training_results.py --push-log && git push
```

## compare_preprocessing (needs DenPAR raw path)

All zeros = **raw DenPAR not at default path**. Fix:

```bash
ls data/DenPAR/Dataset/Training/Key\ Points\ Annotations/ | head
# if missing:
python scripts/compare_preprocessing.py --raw-root /path/to/DenPAR/Dataset
```

Windows: `python scripts/compare_preprocessing.py` works without PYTHONPATH (bootstrap added).

## Not started

- [ ] End-to-end inference + severity ICC
- [ ] Update `paper/replication_progress.tex` after v2/v3 metrics

## Sync to Windows laptop

```bash
git pull   # gets metrics_snapshot.txt, training_results_summary.md
```

Say **"Make a checkpoint"** in chat to refresh this log from new numbers.
