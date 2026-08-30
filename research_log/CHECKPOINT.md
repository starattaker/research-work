# Checkpoint — ICC pairing diagnosed (2026-08-31)

**Status:** Honest test ICC **~0.50–0.57**. Paper **0.801**. Oracle ceiling **~0.79**.

## Completed
- [x] YOLO test mAP50 **0.873** (paper 0.963)
- [x] v6 keypoints: CEJ **0.927** / INT **0.894** / Apex **0.871**
- [x] ICC pipeline + diagnosis (A1=1.0, YOLO not the bottleneck)
- [x] Found GT convention bug: v6 uses **PCA slots**, not paper x-sort (sanity ICC ≈ 0)
- [x] Found pairing bug: pred slot 0 ≠ GT PCA slot 0
- [x] Severity clipped to **[0, 100]** (Eq. 1 is a %)

## Latest friend GPU (2026-08-31)

| Setup | Test ICC | n |
|-------|----------|--:|
| Wrong GT (`paper_x`) | ~0.004 | 875 |
| tensor + PCA GT + slot index | **0.54–0.57** | ~660 |
| lr + PCA + slot index | **0.56** | 659 |
| CEJ-nearest `both_sides` (tensor) | 0.50–0.57 | 662 |
| mask_pca | 0.10 | 609 |
| Train tensor+CEJ | **0.015** | 2155 |
| Val tensor+CEJ | **0.083** | 497 |
| Oracle 8-combo (cheat) | **0.79** | 561 |
| Paper | **0.801** | — |

**Meaning:** keypoints are good enough for ~0.79. ~0.22 ICC is still on **which CEJ/INT/APEX belong together**, plus train/val pairing collapse.

## In progress
- [ ] Hungarian INT/APEX→CEJ assignment (no masks, no GT)
- [ ] Choose protocol on **val**, report **test**
- [ ] One GPU pass per split then CPU sweep

## Not started
- [ ] Retrain intersection (if pairing saturates below ~0.75)
- [ ] External 214-image set
- [ ] Update LaTeX table with ICC

## Resume — friend GPU (one command)

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_icc_friend_full.sh
```
