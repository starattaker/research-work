#!/usr/bin/env bash
# Register existing v1 runs (cej/intersection/apex) without retraining.
set -euo pipefail
REPO="${REPO:-$HOME/faraz/Test_work/research-work}"
cd "$REPO"
git pull origin denpar-severity-replication || true
export PYTHONPATH=.

python - <<'PY'
from pathlib import Path
from src.experiment.registry import after_training, finalize_experiment

for kpt in ("cej", "intersection", "apex"):
    d = Path(f"runs/keypoints/{kpt}")
    if (d / "metrics.json").exists():
        after_training(d, "v1", kpt, auto_push=False)
        print(f"registered v1/{kpt}")
    else:
        print(f"skip missing {d}")

finalize_experiment("v1", auto_push=True)
PY
