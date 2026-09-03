# Checkpoint — ICC sweep done; resume axis (2026-09-04)

**Status:** Friend ICC parameter sweep complete (test ~0.72). Axis script fixed (`ec11fd4`). Resume axis + figures only.

## ICC (v6, just completed on friend)

| Split | ICC | Config |
|-------|----:|--------|
| Val-locked test | **0.7246** | lr + match_by_slot, apex 28 |
| Best test peek | **0.7273** | tensor + match_by_slot, apex 8 |
| Paper | 0.801 | — |

Report: `research_log/icc_parameter_sweep.json`

## Resume on friend (skip ICC GPU re-run)

```bash
cd ~/faraz/Test_work/research-work && git pull origin denpar-severity-replication --no-rebase -X theirs --no-edit && bash scripts/run_paper_resume_axis_friend.sh
```

## Bug fixed
`SideDetail` requires `cej` — fixed in `compare_axis_severity_icc.py` commit `ec11fd4`.
