# ICC — compact notes (2026-09-03)

## Target
Test ICC **0.801**. Honest **~0.73** (v6). Oracle **~0.79**.

## Production config
```
tensor + match_by_slot + apex_merge_px=8 + gt_slot_convention=pca
inference_mode=roi, data=processed_v6, weights=v6_*
```

## Do not use
- `paper_x` GT on processed_v6
- Val-locked hungarian apex28 for test reporting (test drops to ~0.69)
- Oracle 8-combo at inference

## v7
Trained on processed_v7. Intersection OKS **down** vs v6 → ICC unlikely to improve. Run optional check → `icc_v7_report.json`.

## Key files
- `research_log/icc_parameter_sweep.json`
- `research_log/icc_final_report.json`
- `scripts/run_icc_friend_full.sh`
