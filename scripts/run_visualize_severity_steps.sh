#!/usr/bin/env bash
# Find a Python with torch+albumentations (conda or system) and run all pipeline viz steps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

pick_python() {
  local py
  # Project venv (documented in docs/CLONE_AND_TRAIN.md)
  if [[ -x "$(pwd)/venv/bin/python" ]]; then
    py="$(pwd)/venv/bin/python"
    if "$py" -c "import torch, albumentations, cv2, ultralytics" 2>/dev/null; then
      echo "$py"
      return 0
    fi
  fi
  for py in python python3; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import torch, albumentations, cv2, ultralytics" 2>/dev/null; then
      echo "$py"
      return 0
    fi
  done
  for init in \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init"
      conda activate base 2>/dev/null || true
      if python -c "import torch, albumentations, cv2, ultralytics" 2>/dev/null; then
        echo python
        return 0
      fi
    fi
  done
  return 1
}

PYTHON=$(pick_python) || {
  echo "ERROR: No Python with torch, albumentations, opencv, ultralytics found."
  echo "You used '(base)' before — try:  source ~/anaconda3/etc/profile.d/conda.sh && conda activate base"
  echo "Or:  pip install -r requirements.txt   (inside your training venv)"
  exit 1
}

echo "Using: $PYTHON ($($PYTHON --version 2>&1))"
exec "$PYTHON" scripts/visualize_severity_pipeline_steps.py "$@"
