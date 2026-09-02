# Agent handoff — read this first

New session: read in order, then `git log -5` on `denpar-severity-replication`.

| # | File | Purpose |
|---|------|---------|
| 1 | **CHECKPOINT.md** | Current status, numbers, next command |
| 2 | **07_severity_icc.md** | ICC methodology + results (~0.73 test) |
| 3 | **08_axis_severity.md** | PCA vs CEJ–INT axis severity + scripts |
| 4 | **06_keypoint_training.md** | v4–v7 OKS table |
| 5 | **03_preprocessing.md** | v1–v6 preprocess ablations |
| 6 | **ICC_CONTEXT.md** | ICC pitfalls + production config |
| 7 | **paper/replication_progress.tex** | Full draft paper (green = fill after GPU) |

**Branch:** `denpar-severity-replication`  
**Production:** `data/processed_v6`, `runs/keypoints/v6_*`, ICC ~0.73  
**Paper PDF:** `bash scripts/build_paper_pdf.sh` → `paper/replication_progress.pdf`

**Tell the agent:** *"Read `research_log/CHECKPOINT.md` and `research_log/AGENT_HANDOFF.md`."*

## Friend GPU — sync figures to GitHub

```bash
bash scripts/run_paper_figures_friend.sh
# or after manual runs:
bash scripts/push_research_figures.sh
```

## Pending green placeholders in paper

- `axis_severity_icc.json` → Table axis-ICC
- `icc_v7_report.json` → Table v7 ICC
- `paper/figures/axis_severity/*.png` → Fig axis-severity
- 214-image cohort → only after data rights
