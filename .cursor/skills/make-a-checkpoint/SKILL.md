---
name: make-a-checkpoint
description: >-
  Sync research_log/ with the current ML pipeline state (params, counts, run
  status, next steps). Use when the user says "Make a checkpoint", asks to
  update research documentation, or wants research_log refreshed after changes.
---

# Make a checkpoint

Keep `research_log/` accurate, concise, and numeric. No bloat.

## When to run

- User says **"Make a checkpoint"** or asks to update research docs
- After preprocessing, config changes, training start/finish, or evaluation
- Before pausing work so the next session can resume cleanly

## Workflow

1. **Inspect current state** (read only what you need):
   - `research_log/README.md` and `CHECKPOINT.md`
   - Training scripts: `scripts/train_detection.py`, `scripts/train_keypoints.py`, `src/keypoint/train.py`
   - Data summary: `data/processed/preprocess_summary.json`
   - Runs (if any): `runs/detection/`, `runs/keypoints/*/metrics.json`
   - Git diff or recent edits if params/code changed

2. **Update existing docs** when facts changed:
   - `04_hyperparameters.md` — paper value vs **our** value (note VRAM/hardware deviations)
   - `03_preprocessing.md` — counts, paths, re-run commands
   - `01_environment.md` — Python, PyTorch, GPU if changed
   - `02_dataset.md` — split counts if changed

3. **Add or update step docs** (one file per major stage, numbered):
   - `05_detection_training.md` — YOLO run (command, batch, epochs, outputs, metrics)
   - `06_keypoint_training.md` — Keypoint R-CNN runs
   - `07_severity_icc.md` — inference + ICC vs paper target 0.801

4. **Refresh index and resume point**:
   - `README.md` — status table (Done / In progress / Not started)
   - `CHECKPOINT.md` — what's done, what's next, exact resume commands

## Writing rules

- **Informative, not verbose** — tables and numbers over prose
- **Always record**: date, script/command, key hyperparameters, output paths, paper target vs actual (when available)
- **Deviations from paper** must be explicit (e.g. batch 1 vs paper 4 for 4 GB VRAM)
- **Never change** augmentations, LR, optimizer, epochs, or model architecture unless the paper specifies it — only `batch` and `workers` for hardware
- **Do not** start training or change code unless the user asked for that separately

## Paper targets (severity-only scope)

| Stage | Metric | Target |
|-------|--------|--------|
| Tooth detection | mAP50 / mAP50:95 | 0.963 / 0.907 |
| CEJ keypoints | AP50:95 (OKS) | 0.954 |
| Intersection | AP50:95 (OKS) | 0.912 |
| Apex | AP50:95 (OKS) | 0.815 |
| Severity | ICC | 0.801 |

## CHECKPOINT.md template

```markdown
# Checkpoint — [short label] ([YYYY-MM-DD])

**Status:** [Ready to train | Training in progress | Ready for next stage | Blocked]

## Completed
- [x] ...

## In progress
- [ ] ... (path/command)

## Not started
- [ ] ...

## Resume commands
[Exact PowerShell commands for next step only]
```

## After updating

Tell the user briefly:
- Which files were updated
- Current pipeline status
- Next command to run (if any)
