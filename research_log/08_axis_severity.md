# 08 — Axis-constrained severity

**Updated:** 2026-09-03

## Methods

| ID | Axis definition | Module |
|----|-----------------|--------|
| `paper_eq1` | Min-max line through 3 points (sorted x) | `bone_loss.py` |
| `mask_pca` | Mask centroid + PCA major axis | `axis_severity.py` |
| `cej_int_midpoint` | Midpoint(CEJ) → Midpoint(INT) | `axis_severity.py` |

Severity = projected distance ratio along axis, clipped [0,100], geom filter (INT between CEJ and apex).

## Scripts

```bash
# ICC: GT and pred use same axis method, match_by_slot
python scripts/compare_axis_severity_icc.py --split all

# Paper figures (multiple test images)
python scripts/visualize_axis_severity_paper.py --stems 431 5 100 240 622

# Friend one-shot
bash scripts/run_paper_figures_friend.sh
```

Output: `research_log/axis_severity_icc.json`, `paper/figures/axis_severity/`

## Paper table

Fill `paper/replication_progress.tex` Table axis-ICC from JSON after GPU run.
