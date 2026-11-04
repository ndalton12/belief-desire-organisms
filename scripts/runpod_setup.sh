#!/usr/bin/env bash
# Set up a RunPod pod for export, composition, and evaluation. Idempotent: safe to rerun.
# Usually called by scripts/runpod_job.sh; can also be run by hand from a repo checkout.
# Credentials come from the environment (HF_TOKEN, TINKER_API_KEY); never write them to disk.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Persistent volume: model weights, uv cache, pipeline outputs.
# Container disk (local, faster, wiped on stop): the virtualenv, rebuilt quickly from the cache.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/uv-cache}"
export BDO_OUTPUT_DIR="${BDO_OUTPUT_DIR:-$WORKSPACE/outputs}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/bdo-venv}"
export UV_LINK_MODE=copy  # cache and venv are on different filesystems
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$BDO_OUTPUT_DIR"

for var in HF_TOKEN TINKER_API_KEY; do
  if [[ -z "${!var:-}" ]]; then
    echo "warning: $var is not set" >&2
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_DIR"
uv sync --frozen
uv run --frozen python - <<'EOF'
import torch

from belief_desire_organisms.config import load_config

assert torch.cuda.is_available(), "CUDA is not available: check the pod's CUDA version vs torch"
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 2**30
need = load_config().runpod.gpu_memory_gb_min
print(f"GPU: {props.name}, {gib:.0f} GiB, torch {torch.__version__}, CUDA {torch.version.cuda}")
if gib + 2 < need:  # marketing GB vs GiB slack
    raise SystemExit(f"GPU has {gib:.0f} GiB; config asks for at least {need} GB")
EOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
