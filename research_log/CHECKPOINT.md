# Checkpoint — Axis ICC done; paper filled (2026-09-04, updated 2026-09-05)

**Status:** Keypoint training **v6 complete** (2026-09-12 18:44 UTC).

## ICC (v6, completed on friend)

| Split | ICC | Config |
|-------|----:|--------|
| Val-locked test | **0.7246** | lr + match_by_slot, apex 28 |
| Best test peek | **0.7273** | tensor + match_by_slot, apex 8 |
| Paper | 0.801 | — |

Report: `research_log/icc_parameter_sweep.json`

## Axis-constrained severity (v6, end-to-end)

| Split | paper_eq1 | mask_pca | cej_int_midpoint |
|-------|----------:|---------:|-----------------:|
| train | 0.8486 | **0.9319** | 0.8498 |
| val   | 0.7132 | **0.8596** | 0.6844 |
| test  | 0.7260 | **0.8225** | 0.7511 |

Report: `research_log/axis_severity_icc.json`

## Paper

`paper/replication_progress.tex` filled (abstract, ICC table, axis-ICC table,
figures: region-growing, grace sweep, point assignment, axis examples, apex dist).
Build: `pdflatex`/`bibtex` clean (5 pages). Remaining placeholders: v7 counts,
author emails, 214-image cohort.

## Next on friend

Fix GitHub auth then push:
```bash
git remote set-url origin https://USERNAME:TOKEN@github.com/starattaker/research-work.git
git push origin denpar-severity-replication
```

## Latest metrics (auto)

- **v6:** cej OKS=0.843; intersection OKS=0.815; apex OKS=0.781
- Registry: `research_log/experiments/paper_table.json`

