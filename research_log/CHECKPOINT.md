# Checkpoint — v7 complete; paper draft updated (2026-09-03)

**Status:** v6 production ICC ~0.73. Paper draft in `paper/replication_progress.tex`. Run GPU scripts for green placeholders.

## Completed
- [x] v7 all heads trained (v6 still production)
- [x] Full paper LaTeX (intro, methods, tables, green placeholders)
- [x] Axis severity module + ICC script + figure script
- [x] `AGENT_HANDOFF.md`, `08_axis_severity.md`

## v7 vs v6 test OKS

| Model | v6 | v7 | Δ |
|-------|---:|---:|--:|
| CEJ | 0.927 | 0.928 | +0.001 |
| Intersection | 0.894 | 0.882 | −0.012 |
| Apex | 0.871 | 0.881 | +0.010 |

## ICC (v6 production)

Test **~0.73** (tensor + match_by_slot + apex 8px). Paper **0.801**.

## Friend GPU — fill paper placeholders

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_paper_figures_friend.sh
```

## Build PDF

```bash
bash scripts/build_paper_pdf.sh
# Windows: see paper/replication_progress.pdf after MiKTeX build
```

## Pending (green in paper)

- [ ] `axis_severity_icc.json`
- [ ] `icc_v7_report.json`
- [ ] `paper/figures/axis_severity/*.png`
- [ ] 214-image cohort (rights pending)
