# Checkpoint — ICC ~0.73 on test (2026-08-31)

**Status:** Honest end-to-end ICC **~0.70–0.73** on test. Paper **0.801**. Within ~0.07–0.10.

## Completed
- [x] YOLO test mAP50 **0.873** (paper 0.963)
- [x] v6 keypoints: CEJ **0.927** / INT **0.894** / Apex **0.871**
- [x] ICC pipeline fixed (GT convention, pairing, Eq.1 sanity)
- [x] Val-locked sweep (`run_icc_friend_pipeline.py`)

## Latest ICC (friend GPU, 2026-08-31)

| Split | ICC | n pairs | MAE % | Config |
|-------|----:|--------:|------:|--------|
| **Test** | **0.7005** | 597 | 7.4 | hungarian + match_by_slot (val winner) |
| **Test** | **0.7283** | 598 | — | tensor + match_by_slot (best on test) |
| Val | 0.7048 | 449 | 6.4 | hungarian + match_by_slot |
| Train | 0.8279 | 2002 | 4.7 | hungarian + match_by_slot |
| Paper | **0.801** | — | — | — |

## What fixed ICC (0.05 → 0.73)

1. **GT convention** — use `pca` slots from processed_v6 (not `paper_x`).
2. **Pair by slot index** (`match_by_slot`) — GT slot 0 ↔ pred slot 0. CEJ-nearest pairing **hurt** ICC.
3. **Eq.1 sanity** — reject severities when intersection is not between CEJ and apex; clip to **0–100%**.
4. **Combine mode** — `tensor` / `lr` / `hungarian` all ~0.70–0.73; `mask_pca` worse.

## Oracle reference
- Oracle 8-combo (uses GT severity — cheat): **~0.79**
- Gap to paper **0.801** ≈ **0.07** → mostly keypoint + YOLO error, not pairing catastrophe

## In progress
- [ ] Lock production defaults: `tensor` + `match_by_slot` + `pca` GT
- [ ] Report paper-table ICC (train 0.851 / val 0.824 / test 0.801) with same script

## Not started
- [ ] Intersection model improvement (if chasing last ~0.07)
- [ ] External 214-image validation
- [ ] LaTeX table update

## Resume — friend GPU

**ICC parameter grid (combine × protocol × apex_merge):**
```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_icc_optimize_friend.sh
```

**Point assignment / grace sweep (0–48px + outlier analysis):**
```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_point_inclusion_friend.sh
```

**Train v7 (after point inclusion):**
```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_train_v7_from_sweep_friend.sh
```

Legacy full ICC:
```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_icc_friend_full.sh
```

