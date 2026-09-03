#!/usr/bin/env bash
# Sync friend machine — merge pull; on conflict prefer GitHub (theirs).
# Usage:
#   bash scripts/sync_friend_repo.sh
#   SYNC_MODE=reset bash scripts/sync_friend_repo.sh   # discard ALL local commits
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-denpar-severity-replication}"
SYNC_MODE="${SYNC_MODE:-merge}"

# Clean up a stuck merge from a previous failed pull
if [[ -f .git/MERGE_HEAD ]]; then
  echo "Aborting incomplete merge..."
  git merge --abort || true
fi

git fetch origin "$BRANCH"

if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH" "origin/$BRANCH"
elif [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  git checkout "$BRANCH"
fi

if [[ "$SYNC_MODE" = "reset" ]]; then
  git reset --hard "origin/$BRANCH"
else
  # -X theirs: keep GitHub version when research_log auto-commits conflict
  git pull origin "$BRANCH" --no-rebase -X theirs --no-edit
fi

echo "Synced to origin/$BRANCH ($(git rev-parse --short HEAD))"
