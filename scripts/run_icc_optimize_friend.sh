#!/usr/bin/env bash
# ONE command: sync + ICC parameter grid (combine × protocol × apex_merge).
# Optional: SWEEP_YOLO_NMS=1 for score_thresh × nms GPU pass on val.
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

echo ""
echo "Outputs:"
echo "  research_log/icc_parameter_sweep.json"
echo "  recommended: .recommended_defaults in JSON"
