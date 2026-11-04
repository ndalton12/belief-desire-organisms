#!/usr/bin/env bash
# Run one pipeline job on a RunPod GPU pod: fetch the pinned code, set up, fetch inputs from
# the private HF repos, run the job, keep outputs on the persistent volume, upload them to HF,
# then stop the pod (whether the job succeeded or failed).
#
# Usage, inside the pod (as its start command, or over ssh):
#
#   export GIT_COMMIT=<full commit sha> DATASET_REVISION=<dataset repo sha> HF_TOKEN=...
#   export TINKER_API_KEY=...        # export job only
#   curl -fsSL -o /workspace/runpod_job.sh \
#     "https://raw.githubusercontent.com/ndalton12/belief-desire-organisms/$GIT_COMMIT/scripts/runpod_job.sh"
#   bash /workspace/runpod_job.sh <job> [extra options for the job command]
#
# Jobs:
#   baseline  unmodified base model on the dev split: main task and every diagnostic
#   export    download, convert, inspect and parity-check checkpoints of both adapters
#             (EXPORT_CHECKPOINTS="belief-1000000 goal-1000000" to limit; default: all)
#   sweep     every exported checkpoint as a single adapter on the dev split
#   final     base, belief, goal and belief+goal on the test split, with the checkpoints
#             frozen in results/selection.json (fetched from the results repo)
#
# Environment:
#   GIT_COMMIT        required. Full commit SHA of the public repo to run.
#   DATASET_REVISION  required. Commit SHA of the private HF dataset repo (eval files).
#   HF_TOKEN          required. Access to the private dataset, model and results repos.
#   TINKER_API_KEY    required for `export` (downloads and parity sampling are paid calls).
#   GIT_REPO_URL      default https://github.com/ndalton12/belief-desire-organisms.git
#   WORKSPACE         default /workspace (the persistent volume)
#   EXECUTE           default 1. 0 = dry run: no paid calls, no uploads.
#   STOP_POD          default 1: stop this pod at the end. 0 = leave it running (still billed).
#   MAX_JOB_HOURS     default 6. Budget for the whole job from script start (setup, fetches
#                     and every job command); the work is killed when it runs out, then
#                     outputs are still uploaded and the pod still stopped.
#   TRANSFER_TIMEOUT_MIN  default 45. Limit for each HF fetch or upload.
#   DEADLINE_HOURS    default MAX_JOB_HOURS + 2. A detached watchdog stops the pod at this
#                     point no matter what (setup, transfers or the job itself hanging).
#   POD_USD_PER_HOUR  optional. With it, pod time is checked against runpod.max_spend_usd
#                     before starting and recorded in the spend ledger at the end.
#
# Resumption: rerun the same command. Result files skip finished items, export skips
# checkpoints already downloaded or converted, and on a fresh volume earlier uploads are
# fetched back from HF first.
#
# Stopping: RunPod sets RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY in every pod and ships
# runpodctl, so the pod stops itself with `runpodctl pod stop $RUNPOD_POD_ID` (older
# runpodctl: `runpodctl stop pod $RUNPOD_POD_ID`). This runs when the script exits (success,
# failure or kill), and independently from a watchdog at DEADLINE_HOURS. A stopped pod still
# bills for its volume disk.
set -euo pipefail

JOB="${1:-}"
shift || true
case "$JOB" in
  baseline | export | sweep | final) ;;
  *)
    echo "usage: $0 <baseline|export|sweep|final> [extra options]" >&2
    exit 2
    ;;
esac

# Variables given at pod creation reach PID 1 but not always an ssh shell.
if [[ -r /proc/1/environ ]]; then
  while IFS= read -r -d '' kv; do
    name="${kv%%=*}"
    case "$name" in
      RUNPOD_POD_ID | RUNPOD_API_KEY | HF_TOKEN | TINKER_API_KEY | GIT_COMMIT | DATASET_REVISION | \
        POD_USD_PER_HOUR | EXECUTE | STOP_POD | MAX_JOB_HOURS | EXPORT_CHECKPOINTS | \
        TRANSFER_TIMEOUT_MIN | DEADLINE_HOURS)
        if [[ -z "${!name:-}" ]]; then export "$kv"; fi
        ;;
    esac
  done </proc/1/environ
fi

WORKSPACE="${WORKSPACE:-/workspace}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/ndalton12/belief-desire-organisms.git}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/belief-desire-organisms}"
EXECUTE="${EXECUTE:-1}"
STOP_POD="${STOP_POD:-1}"
MAX_JOB_HOURS="${MAX_JOB_HOURS:-6}"
TRANSFER_TIMEOUT_MIN="${TRANSFER_TIMEOUT_MIN:-45}"
DEADLINE_HOURS="${DEADLINE_HOURS:-$(awk "BEGIN {print $MAX_JOB_HOURS + 2}")}"

export WORKSPACE REPO_DIR
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/uv-cache}"
export BDO_OUTPUT_DIR="${BDO_OUTPUT_DIR:-$WORKSPACE/outputs}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/bdo-venv}"
export PATH="$HOME/.local/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=":4096:8" # deterministic cuBLAS (restoration checks are exact)
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$WORKSPACE/logs" "$BDO_OUTPUT_DIR/jobs"
LOG="$WORKSPACE/logs/$JOB-$TS.log"
RECORD="$BDO_OUTPUT_DIR/jobs/$JOB-$TS.json"
exec > >(tee -a "$LOG") 2>&1
START="$(date +%s)"
PY_READY=""
EXEC_FLAG=()
if [[ "$EXECUTE" == 1 ]]; then EXEC_FLAG=(--execute); fi

log() { echo "[job $(date -u +%H:%M:%S)] $*"; }
die() {
  log "ERROR: $*"
  exit 1
}
POD=(uv run --frozen python -m belief_desire_organisms.pod)
# Every HF transfer is time-limited, so a stalled connection cannot keep the pod alive.
transfer() { timeout --kill-after=60 "${TRANSFER_TIMEOUT_MIN}m" "${POD[@]}" "$@"; }

runpodctl_stop() {
  timeout 120 runpodctl pod stop "$RUNPOD_POD_ID" ||
    timeout 120 runpodctl stop pod "$RUNPOD_POD_ID"
}

start_watchdog() {
  if [[ "$STOP_POD" != 1 || -z "${RUNPOD_POD_ID:-}" ]]; then return; fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    log "WARNING: runpodctl not found; no watchdog and no automatic stop"
    return
  fi
  local secs
  secs="$(awk "BEGIN {print int($DEADLINE_HOURS * 3600)}")"
  local cmd="sleep $secs
    echo \"[watchdog] \$(date -u) deadline of $DEADLINE_HOURS h reached; stopping pod\" >>'$LOG'
    timeout 120 runpodctl pod stop '$RUNPOD_POD_ID' || timeout 120 runpodctl stop pod '$RUNPOD_POD_ID'"
  # Detached from this script: it survives a hung or killed job and stops the pod anyway.
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -c "$cmd" >/dev/null 2>&1 &
  else
    nohup bash -c "$cmd" >/dev/null 2>&1 &
  fi
  log "watchdog: pod $RUNPOD_POD_ID will be stopped after $DEADLINE_HOURS h regardless"
}

stop_pod() {
  if [[ "$STOP_POD" != 1 ]]; then
    log "STOP_POD=$STOP_POD: leaving the pod running. It is billed until you stop it."
    return
  fi
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    log "RUNPOD_POD_ID is not set (not a RunPod pod?): nothing to stop"
    return
  fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    log "WARNING: runpodctl not found. Pod $RUNPOD_POD_ID is STILL RUNNING; stop it in the console."
    return
  fi
  sync
  log "stopping pod $RUNPOD_POD_ID"
  runpodctl_stop ||
    log "WARNING: stop failed. Pod $RUNPOD_POD_ID is STILL RUNNING (the watchdog retries at" \
      "$DEADLINE_HOURS h); stop it in the console."
}

finish() {
  local code=$?
  trap - EXIT
  set +e
  log "job '$JOB' ended with exit code $code"
  if [[ -n "$PY_READY" && "$EXECUTE" == 1 ]]; then
    log "uploading outputs to the private HF repos (each limited to $TRANSFER_TIMEOUT_MIN min)"
    transfer upload-results --execute || log "WARNING: results upload failed"
    if [[ "$JOB" == export ]]; then
      transfer upload-adapters --execute || log "WARNING: adapter upload failed"
    fi
  fi
  local minutes=$((($(date +%s) - START) / 60))
  local gpu
  gpu="$(timeout 30 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  printf '{"job": "%s", "started_utc": "%s", "minutes": %d, "exit_code": %d, "git_commit": "%s", "dataset_revision": "%s", "pod_id": "%s", "gpu": "%s", "execute": %s}\n' \
    "$JOB" "$TS" "$minutes" "$code" "${GIT_COMMIT:-}" "${DATASET_REVISION:-}" \
    "${RUNPOD_POD_ID:-}" "$gpu" "$([[ $EXECUTE == 1 ]] && echo true || echo false)" >"$RECORD"
  if [[ -n "$PY_READY" && -n "${POD_USD_PER_HOUR:-}" ]]; then
    timeout 120 uv run --frozen python -c "
from belief_desire_organisms import spend
spend.record('runpod', $minutes / 60 * $POD_USD_PER_HOUR, {'job': '$JOB', 'started_utc': '$TS', 'minutes': $minutes, 'pod_id': '${RUNPOD_POD_ID:-}'})
"
  fi
  log "pod time for this job so far: $minutes min"
  if [[ -n "$PY_READY" && "$EXECUTE" == 1 ]]; then
    if [[ -f "$BDO_OUTPUT_DIR/spend_ledger.jsonl" ]]; then
      transfer upload-file "$BDO_OUTPUT_DIR/spend_ledger.jsonl" --to spend/pod_spend_ledger.jsonl --execute
    fi
    transfer upload-file "$RECORD" --to "jobs/$JOB-$TS.json" --execute
    transfer upload-file "$LOG" --to "logs/$JOB-$TS.log" --execute
  fi
  log "outputs are on the persistent volume under $BDO_OUTPUT_DIR (log: $LOG)"
  stop_pod
  exit "$code"
}
trap finish EXIT
start_watchdog

# --- inputs ---------------------------------------------------------------------------------
[[ "${GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die "GIT_COMMIT must be a full 40-character commit SHA"
[[ -n "${DATASET_REVISION:-}" ]] || die "DATASET_REVISION (dataset repo commit SHA) is required"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is required"
if [[ "$JOB" == export && "$EXECUTE" == 1 && -z "${TINKER_API_KEY:-}" ]]; then
  die "TINKER_API_KEY is required for export"
fi
export BDO_GIT_COMMIT="$GIT_COMMIT"

log "job=$JOB commit=$GIT_COMMIT dataset=$DATASET_REVISION execute=$EXECUTE stop_pod=$STOP_POD"

# --- code at the pinned commit --------------------------------------------------------------
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --quiet "$GIT_REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --quiet origin || true
if ! git -C "$REPO_DIR" cat-file -e "$GIT_COMMIT^{commit}" 2>/dev/null; then
  git -C "$REPO_DIR" fetch --quiet origin "$GIT_COMMIT"
fi
if [[ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]]; then
  die "$REPO_DIR has local changes; refusing to run unpinned code"
fi
git -C "$REPO_DIR" checkout --quiet --detach "$GIT_COMMIT"
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$GIT_COMMIT" ]] || die "checkout did not reach $GIT_COMMIT"

# --- environment ----------------------------------------------------------------------------
bash "$REPO_DIR/scripts/runpod_setup.sh"
cd "$REPO_DIR"
PY_READY=1

if [[ -n "${POD_USD_PER_HOUR:-}" ]]; then
  uv run --frozen python -c "
from belief_desire_organisms import spend
from belief_desire_organisms.config import load_config
spend.check_cap('runpod', load_config().runpod.max_spend_usd, $POD_USD_PER_HOUR * $MAX_JOB_HOURS)
"
fi

# --- inputs from the private HF repos (local files on the volume always win) -----------------
FETCH_DATASET=(--revision "$DATASET_REVISION")
if [[ "$JOB" == export ]]; then FETCH_DATASET+=(--with-train); fi
transfer fetch-dataset "${FETCH_DATASET[@]}"
case "$JOB" in
  export | sweep | final) transfer fetch-model-repo ;;
esac
transfer fetch-results

# --- the job --------------------------------------------------------------------------------
# All job commands share one budget that ends MAX_JOB_HOURS after the script started.
JOB_END="$(awk "BEGIN {print int($START + $MAX_JOB_HOURS * 3600)}")"
run() {
  local left=$((JOB_END - $(date +%s)))
  if ((left <= 0)); then
    log "MAX_JOB_HOURS=$MAX_JOB_HOURS used up; not starting: $*"
    return 124
  fi
  timeout --kill-after=120 "${left}s" "${POD[@]}" "$@"
}

case "$JOB" in
  baseline)
    run baseline "${EXEC_FLAG[@]}" "$@"
    ;;
  export)
    failed=0
    for corpus in belief goal; do
      if [[ ! -f "$BDO_OUTPUT_DIR/training/$corpus/manifest.json" ]]; then
        log "no training manifest for $corpus in the model repo; skipping"
        continue
      fi
      args=()
      for name in ${EXPORT_CHECKPOINTS:-}; do
        if [[ "$name" == "$corpus"-* ]]; then args+=(--checkpoint "$name"); fi
      done
      if [[ -n "${EXPORT_CHECKPOINTS:-}" && ${#args[@]} -eq 0 ]]; then continue; fi
      run export "$corpus" "${args[@]}" "${EXEC_FLAG[@]}" "$@" || failed=1
    done
    [[ "$failed" == 0 ]] || die "export failed for at least one corpus (see log above)"
    ;;
  sweep)
    run sweep "${EXEC_FLAG[@]}" "$@"
    ;;
  final)
    [[ -f "$BDO_OUTPUT_DIR/results/selection.json" ]] ||
      die "results/selection.json not found locally or in the results repo"
    run final "${EXEC_FLAG[@]}" "$@"
    ;;
esac
log "job '$JOB' completed"
