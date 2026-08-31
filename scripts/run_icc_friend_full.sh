#!/usr/bin/env bash
# ONE command for friend GPU: sync repo + full ICC audit & maximize pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
SYNC_MODE=merge bash scripts/sync_friend_repo.sh
source venv/bin/activate
export PYTHONPATH=.
unset RAW_ROOT 2>/dev/null || true
python scripts/run_icc_friend_pipeline.py "$@"
