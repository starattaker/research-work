#!/usr/bin/env bash
# Sync friend machine to remote branch (avoids "divergent branches" pull error).
# Usage:
#   bash scripts/sync_friend_repo.sh              # default: fetch + merge (keeps local commits/files)
#   SYNC_MODE=reset bash scripts/sync_friend_repo.sh   # discard local commits, match remote exactly
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-denpar-severity-replication}"
SYNC_MODE="${SYNC_MODE:-merge}"

git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"

if [ "$SYNC_MODE" = "reset" ]; then
  git reset --hard "origin/$BRANCH"
else
  git merge "origin/$BRANCH" --no-edit
fi

echo "Synced to origin/$BRANCH ($(git rev-parse --short HEAD))"
