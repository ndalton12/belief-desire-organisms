"""Tool versions, code commit and timestamps recorded in export and evaluation manifests."""

from __future__ import annotations

import datetime
import os
import subprocess
from importlib import metadata

from belief_desire_organisms.config import REPO_ROOT

TRACKED_PACKAGES = (
    "belief-desire-organisms",
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "huggingface-hub",
    "tinker",
    "tinker-cookbook",
    "accelerate",
)


def package_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def git_commit() -> str | None:
    """Commit of the code that is running. `BDO_GIT_COMMIT` wins (set by runpod_job.sh)."""
    if os.environ.get("BDO_GIT_COMMIT"):
        return os.environ["BDO_GIT_COMMIT"]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def device_description() -> str:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"{props.name} ({props.total_memory / 2**30:.0f} GiB), CUDA {torch.version.cuda}"
    return "cpu"
