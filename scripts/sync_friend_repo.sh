#!/usr/bin/env bash
# Sync friend machine to remote branch (avoids "divergent branches" pull error).
# Usage:
#   bash scripts/sync_friend_repo.sh              # default: reset to remote (discards local commits)
#   SYNC_MODE=merge bash scripts/sync_friend_repo.sh   # merge remote into local instead
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-denpar-severity-replication}"
SYNC_MODE="${SYNC_MODE:-reset}"

git fetch origin

if [ "$SYNC_MODE" = "merge" ]; then
  git merge "origin/$BRANCH" --no-edit
else
  # Friend GPU: match GitHub exactly (no local merge commits).
  git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
  git reset --hard "origin/$BRANCH"
fi

echo "Synced to origin/$BRANCH ($(git rev-parse --short HEAD))"
