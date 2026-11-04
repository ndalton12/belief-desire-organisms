#!/usr/bin/env bash
# Set up a fresh RunPod pod for export, composition, and evaluation.
# Credentials come from the environment (HF_TOKEN, TINKER_API_KEY); never write them to disk.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/belief-desire-organisms}"

# Keep model weights, datasets, and outputs on persistent storage.
export HF_HOME="$WORKSPACE/hf-cache"
export UV_CACHE_DIR="$WORKSPACE/uv-cache"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

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
uv run python -c "import torch; print('cuda available:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv
