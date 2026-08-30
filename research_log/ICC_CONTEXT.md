# ICC replication — compact context (updated after friend GPU run)

## Paper target
Test ICC **0.801** | train 0.851 | val 0.824

## Critical finding (v6 labels)
- `processed_v6` GT uses **PCA mask slots** at preprocess time, **not** paper x-sort.
- GT sanity: ICC(PCA vs paper_x) ≈ **0** on test → never use `gt_slot_convention=paper_x` with v6.
- **Correct GT**: `--gt-slot-convention pca`

## Best results so far (friend GPU, test)
| Config | ICC | n pairs |
|--------|-----|---------|
| paper_x GT + paper_x pred (wrong) | ~0.004 | 875 |
| **pca GT + tensor + both_sides + roi** | **~0.505** | 663 |
| pca GT + geom_consistent + both_sides | ~0.692 | 108 (few double-side pairs) |
| Oracle 8-combo (uses GT severity — cheat) | ~0.728 | 561 |
| Tensor slot 0/1 (broken pairing) | ~0.059 | 561 |

## Production defaults (after fix)
```bash
--inference-mode roi --combine-mode tensor \
--gt-slot-convention pca --severity-protocol both_sides
```

## NMS
IoU 0.6 on Keypoint R-CNN **detection boxes** (not cross-model keypoint fusion).

## GT severity storage
Not a separate file — computed from `data/processed_v6/keypoints/{cej,intersection,apex}/{split}/annotations/*.json`.

## Gap to 0.801
Oracle ceiling ~0.73 → keypoint error + slot pairing still cost ~0.07–0.08 vs paper. Next: improve intersection keypoints (largest oracle lift).

## One command (friend GPU)
```bash
cd ~/faraz/Test_work/research-work && git pull && source venv/bin/activate && export PYTHONPATH=. && unset RAW_ROOT && bash scripts/run_icc_friend_full.sh
```
