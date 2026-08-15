# Experiment metrics registry

Append-only source for paper tables. **Do not edit by hand** except this README.

| Path | Purpose |
|------|---------|
| `records/` | One JSON per training run: `{timestamp}_{experiment}_{kpt}.json` |
| `registry.json` | Latest OKS per experiment × keypoint; index capped at 200 entries |
| `paper_table.json` | Machine-readable table for all experiments |
| `summaries/paper_table.md` | Human-readable markdown table |
| `summaries/{exp}_keypoints.md` | Per-experiment detail |

Populated automatically by `src/keypoint/train.py` at end of each run. When CEJ + intersection + apex are all recorded for one experiment id, checkpoint + paper fragment update and logs are git-pushed.

Backfill: `python scripts/collect_training_results.py --finalize v1 --push`
