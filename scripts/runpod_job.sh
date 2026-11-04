#!/usr/bin/env bash
# Run one pipeline job unattended on a RunPod GPU pod, from the pod's start command:
# fetch the pinned code, set up, fetch inputs from the private HF repos, run the job, keep
# outputs on the persistent volume, upload them to HF, write a status marker, stop the pod.
#
# Pod start command (one line; the pod needs no ssh and no file transfer):
#
#   bash -c 'curl -fsSL --retry 5 -o /tmp/runpod_job.sh "https://raw.githubusercontent.com/ndalton12/belief-desire-organisms/$GIT_COMMIT/scripts/runpod_job.sh" && exec bash /tmp/runpod_job.sh; echo "could not fetch runpod_job.sh"; [ "$BDO_NO_SELF_STOP" = 1 ] || runpodctl stop pod "$RUNPOD_POD_ID"; sleep infinity'
#
# Image: any RunPod image with bash, git, curl and runpodctl (e.g. a runpod/pytorch image).
# uv installs the locked Python packages; the locked torch is a CUDA 13 build, so the host
# driver must support CUDA 13.0 or newer (choose hosts with CUDA >= 13.0).
#
# Environment (set at pod creation; HF_TOKEN and TINKER_API_KEY preferably as secrets):
#   JOB               required. baseline | export | sweep | final
#   GIT_COMMIT        required. Full commit SHA of the public repo to run.
#   DATASET_REVISION  required. Commit SHA of the private HF dataset repo (eval files).
#   HF_TOKEN          required. Read/write access to the private dataset, model, results repos.
#   TINKER_API_KEY    required for JOB=export (downloads and the paid parity check).
#   JOB_ARGS          optional extra options for the job command, e.g. "--restore-atol 0.05".
#   EXPORT_CHECKPOINTS  JOB=export only: space-separated names, e.g. "belief-0 goal-0".
#   EXECUTE           default 1. 0 = dry run: no paid calls and no result/adapter uploads
#                     (the status marker and log are still uploaded).
#   BDO_NO_SELF_STOP  1 = never stop the pod (neither at the end nor by the watchdog).
#   MAX_JOB_HOURS     default 6. Budget for setup, fetches and the job; the work is killed
#                     when it runs out, then outputs are uploaded and the pod stopped.
#   DEADLINE_HOURS    default MAX_JOB_HOURS + 2. A detached watchdog stops the pod then,
#                     whatever is still running (hung setup, transfer or job).
#   TRANSFER_TIMEOUT_MIN  default 45. Limit for each HF fetch or upload.
#   POD_USD_PER_HOUR  optional. Checks pod time against runpod.max_spend_usd before the job
#                     and records it in the pod's spend ledger afterwards.
#   BDO_RESULTS_REPO  optional. Results repo used before the code is fetched (afterwards
#                     hf.results_repo from configs/pilot.yaml).
#
# Jobs:
#   baseline  unmodified base model on the dev split: main task and every diagnostic
#   export    download, convert, inspect and parity-check checkpoints of both adapters
#   sweep     every exported checkpoint as a single adapter on the dev split
#   final     base, belief, goal and belief+goal on the test split, with the checkpoints
#             frozen in results/selection.json (fetched from the results repo)
#
# Outputs. On the volume: /workspace/outputs (BDO_OUTPUT_DIR), logs in /workspace/logs.
# In the private results repo (cfg.hf.results_repo), on every exit path:
#   status/<job>/<run_id>.json   job, run id, state, exit code, start/end time, commit,
#                                dataset revision, pod id, GPU, uploaded paths. Uploaded last.
#   logs/<job>/<run_id>.log      everything the job printed
#   results/...                  eval runs (paths.results_dir() mirror), when EXECUTE=1
#   spend/<pod_id>/spend_ledger.jsonl   this pod's ledger (parity check, pod time)
# Exported adapters go to adapters/ in the model repo. <run_id> = <pod id>-<UTC time>.
#
# Resumption: create a pod with the same environment. Result files skip finished items,
# export skips checkpoints already downloaded or converted, and on a fresh volume earlier
# uploads are fetched back from HF first.
#
# Stopping: RunPod sets RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY in every pod and ships
# runpodctl, so the pod stops itself (`runpodctl stop pod`, or `runpodctl pod stop` on newer
# runpodctl) after the uploads, on success and on failure. If it cannot, the script sleeps
# instead of exiting so the container is not restarted into a second run. A stopped pod
# still bills for its volume disk.
set -euo pipefail

JOB="${1:-${JOB:-}}"
if [[ $# -gt 0 ]]; then shift; fi

# Variables given at pod creation reach PID 1 but not always other shells.
if [[ -r /proc/1/environ ]]; then
  while IFS= read -r -d '' kv; do
    name="${kv%%=*}"
    case "$name" in
      JOB | RUNPOD_POD_ID | RUNPOD_API_KEY | HF_TOKEN | TINKER_API_KEY | GIT_COMMIT | \
        DATASET_REVISION | JOB_ARGS | EXPORT_CHECKPOINTS | EXECUTE | BDO_NO_SELF_STOP | \
        MAX_JOB_HOURS | DEADLINE_HOURS | TRANSFER_TIMEOUT_MIN | POD_USD_PER_HOUR | \
        BDO_RESULTS_REPO)
        if [[ -z "${!name:-}" ]]; then export "$kv"; fi
        ;;
    esac
  done </proc/1/environ
fi
JOB="${JOB:-}"

WORKSPACE="${WORKSPACE:-/workspace}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/ndalton12/belief-desire-organisms.git}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/belief-desire-organisms}"
EXECUTE="${EXECUTE:-1}"
BDO_NO_SELF_STOP="${BDO_NO_SELF_STOP:-0}"
MAX_JOB_HOURS="${MAX_JOB_HOURS:-6}"
TRANSFER_TIMEOUT_MIN="${TRANSFER_TIMEOUT_MIN:-45}"
DEADLINE_HOURS="${DEADLINE_HOURS:-$(awk "BEGIN {print $MAX_JOB_HOURS + 2}")}"
RESULTS_REPO="${BDO_RESULTS_REPO:-nialldalton12/belief-goal-fruit-pilot-results}"

export WORKSPACE REPO_DIR
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/uv-cache}"
export BDO_OUTPUT_DIR="${BDO_OUTPUT_DIR:-$WORKSPACE/outputs}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/bdo-venv}"
export PATH="$HOME/.local/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=":4096:8" # deterministic cuBLAS (restoration checks are exact)
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

START="$(date +%s)"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID="${RUNPOD_POD_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="${JOB:-unknown}"
mkdir -p "$WORKSPACE/logs/$JOB_NAME" "$BDO_OUTPUT_DIR/status/$JOB_NAME"
LOG="$WORKSPACE/logs/$JOB_NAME/$RUN_ID.log"
STATUS="$BDO_OUTPUT_DIR/status/$JOB_NAME/$RUN_ID.json"
exec > >(tee -a "$LOG") 2>&1

PY_READY=""
UPLOADED=()
EXEC_FLAG=()
if [[ "$EXECUTE" == 1 ]]; then EXEC_FLAG=(--execute); fi
read -r -a EXTRA_ARGS <<<"${JOB_ARGS:-}"

log() { echo "[job $(date -u +%H:%M:%S)] $*"; }
die() {
  log "ERROR: $*"
  exit 1
}

POD=(uv run --frozen python -m belief_desire_organisms.pod)
# Every HF transfer is time-limited, so a stalled connection cannot keep the pod alive.
transfer() { timeout --kill-after=60 "${TRANSFER_TIMEOUT_MIN}m" "${POD[@]}" "$@"; }

# Small-file upload with curl only (HF commit API), for exits before Python is set up.
hf_upload_curl() {
  local src="$1" dest="$2" api="https://huggingface.co/api/datasets/$RESULTS_REPO"
  [[ -n "${HF_TOKEN:-}" ]] || return 1
  local info
  info="$(curl -fsS --max-time 60 -H "Authorization: Bearer $HF_TOKEN" "$api")" || return 1
  if ! grep -q '"private": *true' <<<"$info"; then
    log "refusing to upload to $RESULTS_REPO: it is missing or not private"
    return 1
  fi
  local tmp
  tmp="$(mktemp)"
  tail -c 4000000 "$src" >"$tmp" # keep it a small, regular (non-LFS) file
  printf '{"key":"header","value":{"summary":"upload %s","description":""}}\n{"key":"file","value":{"content":"%s","path":"%s","encoding":"base64"}}\n' \
    "$dest" "$(base64 <"$tmp" | tr -d '\n')" "$dest" |
    curl -fsS --max-time 300 -X POST -H "Authorization: Bearer $HF_TOKEN" \
      -H "Content-Type: application/x-ndjson" --data-binary @- "$api/commit/main" >/dev/null
  local rc=$?
  rm -f "$tmp"
  return $rc
}

upload_file() { # <local path> <path in results repo>
  if [[ -n "$PY_READY" ]]; then
    transfer upload-file "$1" --to "$2" --execute && return 0
  fi
  hf_upload_curl "$1" "$2"
}

write_status() { # <state> <exit code or null>
  local gpu uploaded="" ended=null
  gpu="$(timeout 30 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  if ((${#UPLOADED[@]})); then uploaded="$(printf '"%s",' "${UPLOADED[@]}")"; fi
  if [[ "$1" != running ]]; then ended="\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""; fi
  printf '{"job": "%s", "run_id": "%s", "state": "%s", "exit_code": %s, "started_utc": "%s", "ended_utc": %s, "git_commit": "%s", "dataset_revision": "%s", "pod_id": "%s", "gpu": "%s", "execute": %s, "uploaded": [%s], "log": "logs/%s/%s.log", "volume_outputs": "%s"}\n' \
    "$JOB_NAME" "$RUN_ID" "$1" "$2" "$STARTED_UTC" "$ended" \
    "${GIT_COMMIT:-}" "${DATASET_REVISION:-}" "${RUNPOD_POD_ID:-}" "$gpu" \
    "$([[ $EXECUTE == 1 ]] && echo true || echo false)" "${uploaded%,}" \
    "$JOB_NAME" "$RUN_ID" "$BDO_OUTPUT_DIR" >"$STATUS"
}

runpodctl_stop() {
  timeout 120 runpodctl stop pod "$RUNPOD_POD_ID" ||
    timeout 120 runpodctl pod stop "$RUNPOD_POD_ID"
}

start_watchdog() {
  if [[ "$BDO_NO_SELF_STOP" == 1 || -z "${RUNPOD_POD_ID:-}" ]]; then return; fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    log "WARNING: runpodctl not found; no watchdog and no automatic stop"
    return
  fi
  local secs cmd
  secs="$(awk "BEGIN {print int($DEADLINE_HOURS * 3600)}")"
  cmd="sleep $secs
    echo \"[watchdog] \$(date -u) deadline of $DEADLINE_HOURS h reached; stopping pod\" >>'$LOG'
    timeout 120 runpodctl stop pod '$RUNPOD_POD_ID' || timeout 120 runpodctl pod stop '$RUNPOD_POD_ID'"
  # Detached from this script: it survives a hung or killed job and stops the pod anyway.
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -c "$cmd" >/dev/null 2>&1 &
  else
    nohup bash -c "$cmd" >/dev/null 2>&1 &
  fi
  log "watchdog: pod $RUNPOD_POD_ID will be stopped after $DEADLINE_HOURS h regardless"
}

finish() {
  local code=$?
  trap - EXIT
  set +e
  log "job '$JOB_NAME' ended with exit code $code"
  if [[ -n "$PY_READY" && "$EXECUTE" == 1 ]]; then
    log "uploading outputs (each transfer limited to $TRANSFER_TIMEOUT_MIN min)"
    if transfer upload-results --execute; then
      UPLOADED+=("results/")
    else
      log "WARNING: results upload failed"
    fi
    if [[ "$JOB_NAME" == export ]]; then
      if transfer upload-adapters --execute; then
        UPLOADED+=("model repo: adapters/")
      else
        log "WARNING: adapter upload failed"
      fi
    fi
  fi
  local minutes=$((($(date +%s) - START) / 60))
  if [[ -n "$PY_READY" && -n "${POD_USD_PER_HOUR:-}" ]]; then
    timeout 120 uv run --frozen python -c "
from belief_desire_organisms import spend
spend.record('runpod', $minutes / 60 * $POD_USD_PER_HOUR, {'job': '$JOB_NAME', 'run_id': '$RUN_ID', 'minutes': $minutes})
"
  fi
  local ledger="$BDO_OUTPUT_DIR/spend_ledger.jsonl"
  local ledger_dest="spend/${RUNPOD_POD_ID:-local}/spend_ledger.jsonl"
  if [[ -f "$ledger" ]] && upload_file "$ledger" "$ledger_dest"; then UPLOADED+=("$ledger_dest"); fi
  log "pod time for this job: $minutes min; outputs on the volume under $BDO_OUTPUT_DIR"
  local log_dest="logs/$JOB_NAME/$RUN_ID.log"
  if upload_file "$LOG" "$log_dest"; then UPLOADED+=("$log_dest"); fi
  # The status marker is uploaded last: once it says finished, everything else is done.
  local state=succeeded
  if [[ "$code" != 0 ]]; then state=failed; fi
  write_status "$state" "$code"
  upload_file "$STATUS" "status/$JOB_NAME/$RUN_ID.json" ||
    log "WARNING: could not upload the status marker"
  sync
  if [[ "$BDO_NO_SELF_STOP" == 1 ]]; then
    log "BDO_NO_SELF_STOP=1: leaving the pod running. It is billed until you stop it."
  elif [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    log "RUNPOD_POD_ID is not set (not a RunPod pod): nothing to stop"
    exit "$code"
  elif command -v runpodctl >/dev/null 2>&1; then
    log "stopping pod $RUNPOD_POD_ID"
    runpodctl_stop || log "WARNING: stop failed; the watchdog retries at $DEADLINE_HOURS h"
  else
    log "WARNING: runpodctl not found. Pod $RUNPOD_POD_ID is STILL RUNNING; stop it yourself."
  fi
  # Exiting would make RunPod restart the container and run the job again.
  sleep infinity
}
trap finish EXIT
start_watchdog

# --- inputs ---------------------------------------------------------------------------------
case "$JOB" in
  baseline | export | sweep | final) ;;
  *) die "JOB must be one of baseline, export, sweep, final (got '${JOB}')" ;;
esac
[[ "${GIT_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die "GIT_COMMIT must be a full 40-character commit SHA"
[[ -n "${DATASET_REVISION:-}" ]] || die "DATASET_REVISION (dataset repo commit SHA) is required"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is required"
if [[ "$JOB" == export && "$EXECUTE" == 1 && -z "${TINKER_API_KEY:-}" ]]; then
  die "TINKER_API_KEY is required for export"
fi
export BDO_GIT_COMMIT="$GIT_COMMIT"
log "job=$JOB run_id=$RUN_ID commit=$GIT_COMMIT dataset=$DATASET_REVISION execute=$EXECUTE"
write_status running null
upload_file "$STATUS" "status/$JOB/$RUN_ID.json" || log "WARNING: could not upload the start marker"

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
CONFIG_REPO="$(sed -n 's/^ *results_repo: *\([^ #]*\).*/\1/p' "$REPO_DIR/configs/pilot.yaml" | head -1)"
if [[ -z "${BDO_RESULTS_REPO:-}" && -n "$CONFIG_REPO" && "$CONFIG_REPO" != null ]]; then
  RESULTS_REPO="$CONFIG_REPO"
fi

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
    run baseline "${EXEC_FLAG[@]}" "${EXTRA_ARGS[@]}" "$@"
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
      run export "$corpus" "${args[@]}" "${EXEC_FLAG[@]}" "${EXTRA_ARGS[@]}" "$@" || failed=1
    done
    [[ "$failed" == 0 ]] || die "export failed for at least one corpus (see log above)"
    ;;
  sweep)
    run sweep "${EXEC_FLAG[@]}" "${EXTRA_ARGS[@]}" "$@"
    ;;
  final)
    [[ -f "$BDO_OUTPUT_DIR/results/selection.json" ]] ||
      die "results/selection.json not found locally or in the results repo"
    run final "${EXEC_FLAG[@]}" "${EXTRA_ARGS[@]}" "$@"
    ;;
esac
log "job '$JOB' completed"
