# Checkpoint — v7 complete; v6 production (2026-09-03)

**Status:** v7 trained (all 3 heads). **Use v6 for ICC/paper numbers.** v7 ablation complete — no ICC gain expected.

## Completed
- [x] Improvement sweep → `icc_parameter_sweep.json`, `point_assignment_report.json`
- [x] v7 preprocess: grace **12px**, bbox outlier **7.6px** → `data/processed_v7`
- [x] v7 keypoints trained (CEJ, intersection, apex) — `runs/keypoints/v7_*`

## v7 vs v6 test OKS (`best.pt`)

| Model | v6 | v7 | Δ | Paper |
|-------|---:|---:|--:|------:|
| CEJ | 0.927 | 0.928 | +0.001 | 0.954 |
| Intersection | 0.894 | **0.882** | **−0.012** | 0.912 |
| Apex | 0.871 | 0.881 | +0.010 | 0.815 |
| **Mean** | — | — | mixed | — |

Best val epoch: CEJ ~5, INT ~4, Apex ~5. Heavy overfit (val loss ↑ after epoch 5).

## ICC (production: v6)

| Split | ICC | Config |
|-------|----:|--------|
| Test | **~0.73** | tensor + match_by_slot + apex 8px |
| Val | 0.7338 | parameter grid |
| Paper | **0.801** | — |

**Decision:** v7 does not replace v6. Optional one-shot v7 ICC to confirm (~30 min GPU).

## Resume — v7 ICC check (optional)

```bash
cd ~/faraz/Test_work/research-work && python scripts/run_icc_parameter_sweep.py \
  --data-root data/processed_v6 \
  --cej-weights runs/keypoints/v7_cej/best.pt \
  --intersection-weights runs/keypoints/v7_intersection/best.pt \
  --apex-weights runs/keypoints/v7_apex/best.pt \
  --out research_log/icc_v7_report.json
```

## Paper / handoff

- Progress LaTeX: `paper/replication_progress.tex` (**stale — v4 only; needs v5–v7 + ICC section**)
- Agent onboarding: `research_log/README.md` → `CHECKPOINT.md` → `07_severity_icc.md` → `ICC_CONTEXT.md`

## Not started (minimum for paper)

- [ ] Update `paper/replication_progress.tex` (v6, axis severity, ICC ~0.73)
- [ ] v7 ICC confirm (optional)
- [ ] 1–2 axis-method comparison figures (PCA vs CEJ–INT axis) if pitching axis-constrained severity
