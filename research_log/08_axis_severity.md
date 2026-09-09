# 08 — Axis-constrained severity

**Updated:** 2026-09-04 (GPU run complete; ICC + paper table filled)

## Methods

| ID | Axis definition | Module |
|----|-----------------|--------|
| `paper_eq1` | Min-max line through 3 points (sorted x) | `bone_loss.py` |
| `mask_pca` | Mask centroid + PCA major axis | `axis_severity.py` |
| `cej_int_midpoint` | Midpoint(CEJ) → Midpoint(INT) | `axis_severity.py` |

Severity = projected distance ratio along axis, clipped [0,100], geom filter (INT between CEJ and apex).

## Results (end-to-end, GT method = pred method, match_by_slot)

| Split | paper_eq1 | mask_pca | cej_int_midpoint | GT consistency (paper vs PCA) |
|-------|----------:|---------:|-----------------:|------------------------------:|
| train | 0.8486 (n=2013) | **0.9319** (n=2089) | 0.8498 (n=2001) | 0.807 (MAE 3.12) |
| val   | 0.7132 (n=446) | **0.8596** (n=470) | 0.6844 (n=447) | 0.931 (MAE 1.61) |
| test  | 0.7260 (n=599) | **0.8225** (n=629) | 0.7511 (n=598) | 0.838 (MAE 2.46) |

**Takeaway:** mask-PCA axis beats paper Eq.1 on every split (test +0.097) — it is
the headline number for the paper. Filled `paper/replication_progress.tex` Table
`tab:axis-icc` + abstract + discussion.

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
