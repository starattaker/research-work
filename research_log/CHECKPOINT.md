# Checkpoint — v3/v4 preprocess done (2026-08-20)

**Status:** Keypoint training **v3 complete** (2026-08-20 04:13 UTC).

## Done

- [x] YOLO — test mAP50 **0.873** (paper 0.963)
- [x] v1 CEJ — test OKS **0.820**
- [x] **v2 keypoints** (strict bbox):

| Model | v2 OKS | Paper |
|-------|-------:|------:|
| CEJ | 0.843 | 0.954 |
| Intersection | 0.815 | 0.912 |
| Apex | 0.781 | 0.815 |

- [x] **v3 + v4 preprocess** rebuilt (`data/processed_v3/`, `data/processed_v4/`)
- [x] Comparison stats → `preprocessing_comparison.md` + paper draft updated

## In progress

- [ ] **v3 keypoint training** on friend GPU (first)

## Left

- [ ] v4 keypoint training (after v3)
- [ ] End-to-end inference + ICC (0.801)

## Resume (friend GPU — one step at a time)

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_experiment_v3.sh
```

Use `SKIP_PREPROCESS=1` (data already built). v1/v2 folders untouched.

## Latest metrics (auto)

- **v3:** cej OKS=0.911; intersection OKS=0.817; apex OKS=0.836
- Registry: `research_log/experiments/paper_table.json`

