# Checkpoint — 2026-08-15

**Status:** Automation pushed. v2 ready on friend GPU. v3 in progress on Windows.

## Done

- [x] Repo setup, dataset docs, hyperparameter docs
- [x] Preprocessing **v1 / v2 / v3** code + label stats (`preprocessing_comparison.md`)
- [x] YOLO tooth detection trained — test mAP50 **0.873** (paper 0.963) — see `05_detection_training.md`
- [x] Keypoint R-CNN **CEJ on v1 data** — test OKS **0.820** (paper 0.954) — see `06_keypoint_training.md`
- [x] Training viz, TensorBoard, YOLO/keypoint test scripts
- [x] LaTeX progress paper draft (`paper/replication_progress.tex`)
- [x] **Auto logging** — train end → registry → checkpoint + paper fragment → git push logs

## In progress

- [ ] **v3** — preprocess / keypoint training on Windows (`data/processed_v3/`)
- [ ] **v2** — friend GPU has not run `run_experiment_v2.sh` yet (no v2 metrics in repo)

## Not started / left

- [ ] v1 **intersection + apex** — not documented in repo (friend may have run; needs `register_v1_results.sh` + pull)
- [ ] End-to-end inference (YOLO → keypoints → NMS 0.6)
- [ ] Severity **ICC** vs paper target **0.801**

## Data cross-check (what we actually have in repo)

| Item | Documented? | `metrics.json` / registry in repo? |
|------|-------------|-------------------------------------|
| YOLO test metrics | Yes — `05_detection_training.md` | No weights/metrics git-tracked (`runs/` ignored) |
| CEJ v1 test OKS 0.820 | Yes — `06_keypoint_training.md` | **No** — only prose; friend has `runs/keypoints/cej/metrics.json` locally |
| Intersection v1 | Marked pending in `06` | **No** |
| Apex v1 | Marked pending in `06` | **No** |
| v2 / v3 keypoints | — | **No** |
| Experiment registry | Structure ready | **No records yet** — fills after training + auto-log |

**Paper table source (when runs finish):** `research_log/experiments/paper_table.json`

## Friend — run v2 (one command)

```bash
cd ~/faraz/Test_work/research-work
git pull origin denpar-severity-replication
chmod +x scripts/run_experiment_v2.sh
./scripts/run_experiment_v2.sh
```

Optional if DenPAR path differs:

```bash
RAW_ROOT=/path/to/DenPAR/Dataset ./scripts/run_experiment_v2.sh
```

After v2 finishes, logs auto-push. Here: `git pull`.

## Friend — backfill v1 metrics once (if v1 intersection/apex already trained)

```bash
cd ~/faraz/Test_work/research-work
git pull origin denpar-severity-replication
chmod +x scripts/register_v1_results.sh
./scripts/register_v1_results.sh
```

## Windows — v3 training (after preprocess)

```powershell
cd c:\Oralvis_Seekright
git pull origin denpar-severity-replication
.\scripts\run_experiment_v3.ps1
```
