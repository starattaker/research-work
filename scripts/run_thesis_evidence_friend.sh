#!/usr/bin/env bash
# =============================================================================
# ONE COMMAND: generate every piece of evidence the thesis revision still needs.
#
#   bash scripts/run_thesis_evidence_friend.sh
#
# Run this on the GPU box (the machine that holds runs/keypoints/v6_* and
# data/processed_v6). It is safe to re-run: every stage is idempotent and
# stages that are already done are skipped unless FORCE=1.
#
# What it produces, and why the thesis needs it:
#   Stage 1  v6 metrics.json x3 + registry record
#            -> the production keypoint OKS (0.927/0.894/0.871) currently
#               exists only as prose in a markdown log. This gives it an
#               audit trail.
#   Stage 2  per-tooth severity pair CSVs
#            -> only aggregates were ever saved; the raw pairs were discarded.
#   Stage 3  ICC bootstrap CIs + Bland-Altman limits of agreement
#            -> the examiner asked for a numerical stability analysis.
#
# It pushes after EVERY stage, so partial progress is never lost and the
# Windows machine can pull results while later stages are still running.
#
# Env overrides:
#   BRANCH=...      git branch            (default denpar-severity-replication)
#   DATA_ROOT=...   processed data root   (default data/processed_v6)
#   EXP=...         experiment id         (default v6)
#   DEVICE=...      cuda | cpu            (default cuda)
#   NBOOT=...       bootstrap resamples   (default 2000)
#   FORCE=1         redo stages even if their outputs already exist
#   SKIP_STAGES=... comma list, e.g. SKIP_STAGES=1,2
# =============================================================================

set -uo pipefail   # deliberately NOT -e: one failing stage must not kill the run
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BRANCH="${BRANCH:-denpar-severity-replication}"
DATA_ROOT="${DATA_ROOT:-data/processed_v6}"
EXP="${EXP:-v6}"
DEVICE="${DEVICE:-cuda}"
NBOOT="${NBOOT:-2000}"
FORCE="${FORCE:-0}"
SKIP_STAGES="${SKIP_STAGES:-}"

LOG_DIR="research_log/run_logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/thesis_evidence_$(date -u +%Y%m%dT%H%M%SZ).log"

FAILED_STAGES=()
DONE_STAGES=()

# ---------------------------------------------------------------- helpers ---
log()  { printf '%s\n' "$*" | tee -a "$RUN_LOG"; }
hdr()  { log ""; log "=============================================================="; log "$*"; log "=============================================================="; }
warn() { printf '\033[33mWARN\033[0m %s\n' "$*" | tee -a "$RUN_LOG"; }
err()  { printf '\033[31mFAIL\033[0m %s\n' "$*" | tee -a "$RUN_LOG"; }

skipped() { [[ ",$SKIP_STAGES," == *",$1,"* ]]; }

# Commit + push whatever the stage produced. Never aborts the run.
push_stage() {
  local msg="$1"; shift
  local paths=("$@")
  local existing=()
  for p in "${paths[@]}"; do [[ -e "$p" ]] && existing+=("$p"); done
  if [[ ${#existing[@]} -eq 0 ]]; then
    warn "push: nothing to stage for '$msg'"
    return 0
  fi

  git add -- "${existing[@]}" 2>/dev/null || true
  if git diff --cached --quiet; then
    log "push: no changes to commit for '$msg'"
    return 0
  fi
  git commit -m "$msg" >>"$RUN_LOG" 2>&1 || { warn "push: commit failed"; return 0; }

  # Up to 3 attempts; on rejection, merge remote keeping OUR fresh results.
  for attempt in 1 2 3; do
    if git push origin "$BRANCH" >>"$RUN_LOG" 2>&1; then
      log "pushed: $msg"
      return 0
    fi
    warn "push rejected (attempt $attempt) - merging remote and retrying"
    git merge --abort 2>/dev/null || true
    git pull origin "$BRANCH" --no-rebase --no-edit -X ours >>"$RUN_LOG" 2>&1 || true
  done
  warn "push failed after 3 attempts - commit is saved locally, push manually later"
  return 0
}

run_stage() {
  local num="$1" name="$2"; shift 2
  if skipped "$num"; then log "-- stage $num ($name) skipped by SKIP_STAGES"; return 0; fi
  hdr "STAGE $num - $name"
  if "$@"; then
    DONE_STAGES+=("$num:$name")
    return 0
  fi
  err "stage $num ($name) did not complete"
  FAILED_STAGES+=("$num:$name")
  return 1
}

# ------------------------------------------------------------ environment ---
hdr "ENVIRONMENT"
log "repo:    $REPO_ROOT"
log "branch:  $BRANCH"
log "data:    $DATA_ROOT"
log "device:  $DEVICE"
log "log:     $RUN_LOG"

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
  log "venv:    activated"
else
  warn "no venv/bin/activate - using system python3"
fi
export PYTHONPATH="${PYTHONPATH:-.}"
PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
log "python:  $($PY --version 2>&1)"

$PY - <<'PYCHECK' 2>&1 | tee -a "$RUN_LOG"
try:
    import torch
    print(f"torch:   {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu:     {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch:   NOT IMPORTABLE ({exc})")
PYCHECK

# --------------------------------------------------------------- stage 0 ----
hdr "STAGE 0 - sync with GitHub"
if [[ -f .git/MERGE_HEAD ]]; then
  warn "clearing a stuck merge from a previous run"
  git merge --abort 2>/dev/null || true
fi
git fetch origin "$BRANCH" >>"$RUN_LOG" 2>&1 || warn "fetch failed (offline?) - continuing with local state"
if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH" "origin/$BRANCH" >>"$RUN_LOG" 2>&1 || warn "checkout failed"
elif [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  git checkout "$BRANCH" >>"$RUN_LOG" 2>&1 || warn "checkout failed"
fi
git pull origin "$BRANCH" --no-rebase --no-edit -X theirs >>"$RUN_LOG" 2>&1 \
  || warn "pull failed - continuing with local state"
log "at commit: $(git rev-parse --short HEAD)"

# ------------------------------------------------------------ preflight -----
hdr "PREFLIGHT - checking prerequisites"
PREFLIGHT_OK=1
for w in "runs/keypoints/${EXP}_cej/best.pt" \
         "runs/keypoints/${EXP}_intersection/best.pt" \
         "runs/keypoints/${EXP}_apex/best.pt"; do
  if [[ -f "$w" ]]; then log "  OK      $w"; else err "  MISSING $w"; PREFLIGHT_OK=0; fi
done
if [[ -d "$DATA_ROOT" ]]; then log "  OK      $DATA_ROOT"; else err "  MISSING $DATA_ROOT"; PREFLIGHT_OK=0; fi
if compgen -G "runs/detect*/**/best.pt" >/dev/null 2>&1 || [[ -f runs/detect/runs/detection/yolov8x_tooth/weights/best.pt ]]; then
  log "  OK      YOLO detection weights"
else
  warn "  MISSING YOLO weights - stage 2 end-to-end will fail (GT-only still works)"
fi

if [[ "$PREFLIGHT_OK" -eq 0 ]]; then
  err "Prerequisites missing. This script must run on the machine holding the"
  err "v6 checkpoints and $DATA_ROOT. Nothing was changed."
  exit 1
fi

# --------------------------------------------------------------- stage 1 ----
stage1_v6_metrics() {
  local ok=1
  for kpt in cej intersection apex; do
    local out_dir="runs/keypoints/${EXP}_${kpt}"
    if [[ -f "$out_dir/metrics.json" && "$FORCE" != "1" ]]; then
      log "  metrics.json already present for $kpt (FORCE=1 to redo)"
      continue
    fi
    log "  evaluating $kpt ..."
    if $PY -m src.keypoint.train \
        --data-root "$DATA_ROOT/keypoints/$kpt" \
        --keypoint-type "$kpt" \
        --output-dir "$out_dir" \
        --experiment-id "$EXP" \
        --device "$DEVICE" \
        --eval-only --no-auto-push >>"$RUN_LOG" 2>&1; then
      log "  wrote $out_dir/metrics.json"
    else
      err "  eval failed for $kpt (see $RUN_LOG)"
      ok=0
    fi
  done

  log "  updating experiment registry ..."
  $PY scripts/collect_training_results.py --experiment-id "$EXP" --no-push >>"$RUN_LOG" 2>&1 \
    || warn "  registry backfill failed"

  push_stage "Add ${EXP} keypoint metrics.json + registry record (thesis audit trail)" \
    research_log/experiments runs/keypoints/*/metrics.json
  [[ "$ok" -eq 1 ]]
}

# --------------------------------------------------------------- stage 2 ----
stage2_dump_pairs() {
  local out_dir="research_log/severity_pairs"
  if [[ -f "$out_dir/severity_pairs_endtoend.csv" && "$FORCE" != "1" ]]; then
    log "  severity pair CSVs already present (FORCE=1 to redo)"
  else
    $PY scripts/dump_severity_pairs.py \
      --data-root "$DATA_ROOT" \
      --split all \
      --device "$DEVICE" \
      --cej-weights "runs/keypoints/${EXP}_cej/best.pt" \
      --intersection-weights "runs/keypoints/${EXP}_intersection/best.pt" \
      --apex-weights "runs/keypoints/${EXP}_apex/best.pt" \
      --out-dir "$out_dir" 2>&1 | tee -a "$RUN_LOG"
    [[ "${PIPESTATUS[0]}" -eq 0 ]] || { err "  pair dump failed"; return 1; }
  fi
  push_stage "Add per-tooth severity pair CSVs (enables CI + Bland-Altman)" "$out_dir"
}

# --------------------------------------------------------------- stage 3 ----
stage3_agreement() {
  $PY scripts/analyze_agreement.py \
    --pairs-dir research_log/severity_pairs \
    --out research_log/severity_agreement.json \
    --fig-dir research_log/figures/agreement \
    --n-boot "$NBOOT" 2>&1 | tee -a "$RUN_LOG"
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || { err "  agreement analysis failed"; return 1; }
  push_stage "Add ICC bootstrap CIs + Bland-Altman agreement analysis" \
    research_log/severity_agreement.json research_log/figures/agreement
}

# ------------------------------------------------------------------ run -----
run_stage 1 "v6 keypoint metrics + registry"      stage1_v6_metrics
run_stage 2 "dump per-tooth severity pairs"       stage2_dump_pairs
run_stage 3 "ICC CIs + Bland-Altman"              stage3_agreement

# Always push the run log itself so the Windows box can read what happened.
push_stage "Add thesis-evidence run log" "$RUN_LOG"

# -------------------------------------------------------------- summary -----
hdr "SUMMARY"
for s in "${DONE_STAGES[@]:-}";   do [[ -n "$s" ]] && log "  DONE   $s"; done
for s in "${FAILED_STAGES[@]:-}"; do [[ -n "$s" ]] && err "  FAILED $s"; done

log ""
log "Artifacts produced:"
for f in runs/keypoints/${EXP}_cej/metrics.json \
         runs/keypoints/${EXP}_intersection/metrics.json \
         runs/keypoints/${EXP}_apex/metrics.json \
         research_log/experiments/registry.json \
         research_log/severity_pairs/severity_pairs_gtonly.csv \
         research_log/severity_pairs/severity_pairs_endtoend.csv \
         research_log/severity_agreement.json; do
  [[ -f "$f" ]] && log "  + $f" || log "  - $f (not produced)"
done
log ""
log "Full log: $RUN_LOG"

if [[ ${#FAILED_STAGES[@]} -gt 0 ]]; then
  err "Finished with ${#FAILED_STAGES[@]} failed stage(s). Everything that DID"
  err "succeed has been committed and pushed."
  exit 1
fi
log "All stages completed and pushed to origin/$BRANCH."
