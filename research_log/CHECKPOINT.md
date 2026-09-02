# Checkpoint — v7 train partial (2026-09-02)

**Status:** v7 CEJ + intersection done; apex interrupted epoch 2. **Use v6 for ICC** until apex done + ICC re-run.

## v7 preprocess (`processed_v7`)

`max_grace_px=12`, `bbox_outlier_margin_px=7.6` — 4401 teeth (1 skipped no mask).

## v7 keypoint OKS (test, `best.pt`)

| Model | v6 | v7 | Δ | Best val epoch |
|-------|---:|---:|--:|----------------|
| CEJ | 0.927 | **0.928** | +0.001 | ~5 |
| Intersection | 0.894 | **0.882** | −0.012 | ~4 |
| Apex | 0.871 | — | — | interrupted |

Severe overfit: train loss ↓, val loss ↑ after epoch ~5–6. Early stopping used `best.pt` correctly.

## Resume apex only

```bash
cd ~/faraz/Test_work/research-work && SKIP_PREPROCESS=1 SKIP_DONE=1 bash scripts/run_train_v7_from_sweep_friend.sh
```

## ICC (v6, still production)

Honest test **~0.73** (`tensor + match_by_slot + apex 8px`). v7 unlikely to beat v6 on intersection.
