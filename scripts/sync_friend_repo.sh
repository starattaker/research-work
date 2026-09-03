#!/usr/bin/env bash
# Sync friend machine — merge pull (never hard-reset unless SYNC_MODE=reset).
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-denpar-severity-replication}"
SYNC_MODE="${SYNC_MODE:-merge}"

git fetch origin "$BRANCH"

if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH" "origin/$BRANCH"
elif [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  git checkout "$BRANCH"
fi

if [[ "$SYNC_MODE" = "reset" ]]; then
  git reset --hard "origin/$BRANCH"
else
  # Avoid "Need to specify how to reconcile divergent branches" on modern git
  git pull origin "$BRANCH" --no-rebase --no-edit
fi

echo "Synced to origin/$BRANCH ($(git rev-parse --short HEAD))"
