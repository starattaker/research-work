#!/usr/bin/env bash
# Serial: merge-pull → ICC parameter grid → point-assignment sweep (no training).
set -euo pipefail
cd "$(dirname "$0")/.."
SYNC_MODE=merge bash scripts/sync_friend_repo.sh
source venv/bin/activate
export PYTHONPATH=.
unset RAW_ROOT 2>/dev/null || true

EXTRA=()
if [[ "${SWEEP_YOLO_NMS:-0}" == "1" ]]; then
  EXTRA+=(--sweep-yolo-nms)
fi

python scripts/run_icc_parameter_sweep.py "${EXTRA[@]}" "$@"

DEFAULT_RAW="data/DenPAR/Dataset"
if [[ -z "${RAW_ROOT:-}" ]] || [[ ! -d "${RAW_ROOT}/Testing/Key Points Annotations" ]]; then
  RAW_ROOT="$DEFAULT_RAW"
fi
python scripts/analyze_point_assignment_full.py \
  --raw-root "$RAW_ROOT" \
  --split all \
  --max-radius "${MAX_RADIUS:-48}"
python scripts/analyze_grace_radius_sweep.py \
  --raw-root "$RAW_ROOT" \
  --split Testing \
  --max-radius 24
python scripts/analyze_apex_merge_radius.py \
  --data-root data/processed_v6 \
  --split all
