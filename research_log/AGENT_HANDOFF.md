# Agent handoff — read this first

New session: read in order, then `git log -5` on `denpar-severity-replication`.

| # | File | Purpose |
|---|------|---------|
| 1 | **CHECKPOINT.md** | Current status, numbers, next command |
| 2 | **07_severity_icc.md** | ICC methodology + results |
| 3 | **06_keypoint_training.md** | v4–v7 OKS table |
| 4 | **03_preprocessing.md** | v1–v6 preprocess ablations |
| 5 | **ICC_CONTEXT.md** | ICC pitfalls + production config |
| 6 | **preprocessing_comparison.md** | Region-growing stats (multipass pitch) |
| 7 | **paper/replication_progress.tex** | Draft paper (needs update to v6/ICC) |

**Branch:** `denpar-severity-replication`  
**Production weights:** `runs/keypoints/v6_*`, YOLO `runs/detect/.../best.pt`  
**Production data:** `data/processed_v6`  
**Honest test ICC:** ~0.73 (paper 0.801)

**Tell the agent:** *"Read `research_log/CHECKPOINT.md` and `research_log/ICC_CONTEXT.md`, then continue from there."*
