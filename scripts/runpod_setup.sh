#!/usr/bin/env bash
# Set up a RunPod pod for export, composition, and evaluation. Idempotent: safe to rerun.
# Usually called by scripts/runpod_job.sh; can also be run by hand from a repo checkout.
# Credentials come from the environment (HF_TOKEN, TINKER_API_KEY); never write them to disk.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Persistent volume: pipeline outputs only (results, exported adapters), so a stopped job
# resumes. Container disk (local, wiped on stop): the virtualenv and both download caches.
# The caches are set explicitly, ignoring image defaults (runpod/pytorch presets a cache
# under /workspace): some hosts' volumes reject the renames uv and huggingface_hub use
# ("Cross-device link"). Re-downloading on a fresh container costs a few minutes.
export HF_HOME="${BDO_HF_HOME:-/root/.cache/bdo-hf}"
export UV_CACHE_DIR="${BDO_UV_CACHE_DIR:-/root/.cache/bdo-uv}"
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
gb = props.total_memory / 1e9  # decimal GB, the unit cards are sold in
need = load_config().runpod.gpu_memory_gb_min
print(f"GPU: {props.name}, {gb:.1f} GB, torch {torch.__version__}, CUDA {torch.version.cuda}")
# A "48 GB" A40 reports ~47.7 GB (44.4 GiB) usable, so allow 10% below the nominal size.
if gb < 0.9 * need:
    raise SystemExit(f"GPU has {gb:.1f} GB; config asks for a {need} GB card")
EOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
