# Checkpoint — ICC pairing + severity guards (2026-08-31)

**Status:** Honest test ICC **~0.50–0.57**. Paper **0.801**. Oracle ceiling **~0.79**.

## Completed

- [x] YOLO test mAP50 **0.873** (paper 0.963)
- [x] Keypoints v6: CEJ **0.927** / INT **0.894** / Apex **0.871**
- [x] ICC pipeline: YOLO ROI + 3× Keypoint R-CNN + Eq. 1
- [x] Diagnosed low ICC: **not YOLO**. Main issues were **wrong GT convention** (paper_x vs v6 PCA) and **slot-index pairing**
- [x] Friend GPU audit 2026-08-31 (commit `7e0783d` era)

## Latest ICC numbers (friend GPU, 2026-08-31)

| Setup | Test ICC | n | Notes |
|-------|----------|--:|-------|
| paper_x GT (wrong) | ~0.00 | 875 | v6 labels are PCA slots |
| tensor + pca GT + slot-index | **0.54–0.57** | ~660 | best honest high-n |
| lr + pca + slot-index | 0.56 | 659 | |
| tensor + CEJ-match (STEP 3) | 0.50 | 662 | CEJ match can hurt |
| mask_pca preds | 0.10 | 609 | axis sign ≠ GT |
| Oracle 8-combo (uses GT sev) | **0.79** | 561 | not for publication |
| **Paper** | **0.801** | — | |

Train/val ICC **0.01–0.17** with same models — likely unbounded / cross-side Eq. 1 outliers. Guards now: clip **[0, 100]** + drop INT not between CEJ and apex.

## In progress

- [ ] Hungarian INT/APEX→CEJ assignment + clip/between-axis (this push)
- [ ] Get **train ≈ val ≈ test** ICC (if train stays ~0, pairing/outliers still broken)

## Not started

- [ ] Retrain intersection (largest remaining error vs paper)
- [ ] Team 214-image external set

## Resume — friend GPU (one command)

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_icc_friend_full.sh
```

Extracts test keypoints once, sweeps hungarian/tensor/lr/mask_pca × pairing, then train/val/test on the winner. Report: `research_log/icc_final_report.json`
