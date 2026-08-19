# Checkpoint — 2026-08-20

**Status:** Your v3 training failed because the wrong Python was used (not the project venv). Fix below, then re-run v3.

## The project in one sentence

We are copying a dental X-ray paper: find teeth (YOLO) → find 3 keypoint types (CEJ, intersection, apex) → later score severity (ICC).

## Two machines

| Machine | Job |
|---------|-----|
| **Friend GPU (Linux)** | Heavy training — v1 done partly, **v2 not started yet** |
| **Your laptop (Windows)** | v3 preprocess + v3 keypoint training |

## Done

- Preprocessing code (v1, v2, v3)
- YOLO trained on friend machine — mAP50 **0.873**
- CEJ keypoints on v1 data — OKS **0.820** (friend machine)
- Auto logging code pushed to GitHub

## On friend machine (we think)

- YOLO — done
- CEJ v1 — done
- Intersection + apex v1 — **unknown** (not synced to GitHub yet)
- v2 experiment — **not started**

## On your laptop

- v3 preprocess — likely done (`data/processed_v3/`)
- v3 training — **failed** (venv not activated)

## Left (whole project)

- Finish v3 keypoints (you)
- Run v2 on friend GPU
- Final pipeline + ICC score

## `register_v1_results.sh` — ignore for now

Optional one-time script **only on friend machine** if intersection/apex were already trained there. You do not need it on Windows.

## Data in GitHub repo

Only **docs + small JSON logs** sync via git. Model weights stay on each machine (`runs/`). Registry fills automatically when training succeeds.
