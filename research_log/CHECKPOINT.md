# Checkpoint — ICC ~0.73 on test (2026-08-31)

**Status:** Ready for parameter sweep + v7 preprocess. Test ICC **~0.73** (paper **0.801**).

## Completed
- [x] YOLO test mAP50 **0.873** (paper 0.963)
- [x] v6 keypoints: CEJ **0.927** / INT **0.894** / Apex **0.871**
- [x] ICC pipeline fixed (PCA GT, match_by_slot, Eq.1 clip + geom filter)
- [x] Val-locked sweep `861dfca` → `research_log/icc_final_report.json`
- [x] Friend scripts: ICC grid, point-assignment sweep, v7 train (`402e0bc`)

## Latest ICC (friend GPU, 2026-08-31, commit `861dfca`)

| Split | ICC | n | MAE % | Config |
|-------|----:|--:|------:|--------|
| Test (val-locked) | **0.7005** | 597 | 7.4 | hungarian + match_by_slot |
| Test (best on test) | **0.7283** | 598 | — | tensor + match_by_slot |
| Val | 0.7048 | 449 | 6.4 | hungarian + match_by_slot |
| Train | 0.8279 | 2002 | 4.7 | hungarian + match_by_slot |
| Paper | **0.801** | — | — | — |

Oracle 8-combo ~**0.79** (ceiling with GT pairing cheat).

## In progress
- [ ] ICC parameter grid: combine × protocol × apex_merge_px
- [ ] Grace sweep 0–48px + bbox outlier → v7 preprocess

## Not started
- [ ] v7 keypoint retrain (`processed_v7`)
- [ ] External 214-image validation

## Resume — friend GPU (merge pull, keeps local files)

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_improvement_friend.sh
```

After that finishes:

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_train_v7_from_sweep_friend.sh
```

Manual merge only:

```bash
cd ~/faraz/Test_work/research-work && git fetch origin && git checkout denpar-severity-replication && git merge origin/denpar-severity-replication --no-edit
```

Outputs: `research_log/icc_parameter_sweep.json`, `research_log/figures/point_assignment_full/`
