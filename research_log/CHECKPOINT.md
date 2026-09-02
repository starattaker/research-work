# Checkpoint — v7 train partial (2026-09-02)

**Status:** Keypoint training **v7 complete** (2026-09-02 01:29 UTC).

## Completed
- [x] Improvement sweep (`icc_parameter_sweep.json`, point assignment)
- [x] v7 preprocess: grace **12px**, bbox outlier **7.6px** → `processed_v7`
- [x] v7 CEJ + intersection trained (`best.pt`)

## v7 keypoint OKS (test)

| Model | v6 | v7 | Δ | Best val epoch |
|-------|---:|---:|--:|----------------|
| CEJ | 0.927 | **0.928** | +0.001 | ~5 |
| Intersection | 0.894 | **0.882** | −0.012 | ~4 |
| Apex | 0.871 | — | — | interrupted @ ep 2 |

Overfit: val loss best ~epoch 4–5, then rose to 8+ by epoch 35. Early stopping kept `best.pt`.

## ICC (v6, production)

Honest test **~0.73** (`tensor + match_by_slot + apex 8px`). Paper **0.801**.

## Resume apex only

```bash
cd ~/faraz/Test_work/research-work && git fetch origin && git merge origin/denpar-severity-replication --no-edit && SKIP_PREPROCESS=1 SKIP_DONE=1 bash scripts/run_train_v7_from_sweep_friend.sh
```

## Latest metrics (auto)

- **v7:** cej OKS=0.928; intersection OKS=0.882; apex OKS=0.881
- Registry: `research_log/experiments/paper_table.json`

