"""Private Hugging Face repositories used by export and evaluation.

Two repositories besides the dataset repo (spec section 4):

Model repo (`cfg.hf.model_repo`, repo_type "model"); `adapters/` mirrors `adapters_dir()`::

    manifests/training_<corpus>.json                TrainingManifest (pushed after training)
    adapters/<corpus>/export_manifest.json          ExportManifest: tool versions, base revision
    adapters/<corpus>/parity.json                   local vs Tinker log-probability comparison
    adapters/<corpus>/<checkpoint>/tinker/          original Tinker download
    adapters/<corpus>/<checkpoint>/peft/            converted PEFT adapter
    adapters/<corpus>/<checkpoint>/export.json      inspection record for that checkpoint

Results repo (`cfg.hf.results_repo`, repo_type "dataset"), mirrors `results_dir()`::

    results/<split>/<condition>/<checkpoint_label>.jsonl      one EvalResult per item
    results/<split>/<condition>/<checkpoint_label>.run.json   run metadata and checks
    results/selection.json                frozen checkpoint choice for the final run
    jobs/<job>-<timestamp>.json           pod job records (duration, status, commit)
    logs/<job>-<timestamp>.log            pod job logs

Every write is behind `execute`. A repository is created private when missing, and an
existing repository that is not private is refused. Downloads never overwrite a local file:
the pod volume is the primary copy, Hugging Face is the backup.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from belief_desire_organisms.conditions import Corpus

MODEL_REPO_TYPE = "model"
RESULTS_REPO_TYPE = "dataset"
DATASET_REPO_TYPE = "dataset"

MODEL_REPO_ADAPTERS_DIR = "adapters"
RESULTS_REPO_RESULTS_DIR = "results"
RESULTS_REPO_JOBS_DIR = "jobs"
RESULTS_REPO_LOGS_DIR = "logs"

# Never uploaded: unfinished downloads/conversions and Python caches.
UPLOAD_IGNORE = ["**/*.partial/**", "*.partial/**", "**/.download-*/**", "**/__pycache__/**"]


def training_manifest_path_in_repo(corpus: Corpus) -> str:
    return f"manifests/training_{corpus.value}.json"


class RepoNotPrivateError(RuntimeError):
    pass


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def ensure_private_repo(repo_id: str, repo_type: str, api=None) -> None:
    """Create the repo as private if missing; refuse if it exists and is public."""
    from huggingface_hub.errors import RepositoryNotFoundError

    api = api or _api()
    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        info = api.repo_info(repo_id, repo_type=repo_type)
    if not getattr(info, "private", False):
        raise RepoNotPrivateError(
            f"{repo_type} repo {repo_id} is not private; refusing to upload. "
            "Make it private on Hugging Face first."
        )


def upload_folder(
    repo_id: str | None,
    repo_type: str,
    folder: Path,
    path_in_repo: str,
    *,
    execute: bool,
    commit_message: str,
    allow_patterns: list[str] | None = None,
    api=None,
) -> bool:
    """Upload `folder` to `path_in_repo`. Returns True if an upload happened."""
    if repo_id is None:
        print(f"[hf] no repo configured; not uploading {folder}")
        return False
    if not folder.exists():
        print(f"[hf] nothing to upload: {folder} does not exist")
        return False
    if not execute:
        print(f"[hf] dry run: would upload {folder} -> {repo_type}:{repo_id}/{path_in_repo}")
        return False
    api = api or _api()
    ensure_private_repo(repo_id, repo_type, api=api)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        allow_patterns=allow_patterns,
        ignore_patterns=UPLOAD_IGNORE,
    )
    print(f"[hf] uploaded {folder} -> {repo_type}:{repo_id}/{path_in_repo}")
    return True


def upload_file(
    repo_id: str | None,
    repo_type: str,
    local_path: Path,
    path_in_repo: str,
    *,
    execute: bool,
    commit_message: str,
    api=None,
) -> bool:
    if repo_id is None:
        print(f"[hf] no repo configured; not uploading {local_path}")
        return False
    if not execute:
        print(f"[hf] dry run: would upload {local_path} -> {repo_type}:{repo_id}/{path_in_repo}")
        return False
    api = api or _api()
    ensure_private_repo(repo_id, repo_type, api=api)
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    print(f"[hf] uploaded {local_path} -> {repo_type}:{repo_id}/{path_in_repo}")
    return True


def download_file(
    repo_id: str,
    repo_type: str,
    filename: str,
    local_path: Path,
    *,
    revision: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download one file to an exact local path. Keeps an existing local file unless told."""
    from huggingface_hub import hf_hub_download

    if local_path.exists() and not overwrite:
        return local_path
    with tempfile.TemporaryDirectory() as tmp:
        got = hf_hub_download(
            repo_id, filename, repo_type=repo_type, revision=revision, local_dir=tmp
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, local_path)
    return local_path


def download_missing(
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    local_dir: Path,
    *,
    revision: str | None = None,
) -> list[Path]:
    """Download files under `path_in_repo` that are missing from `local_dir`.

    `local_dir` corresponds to `path_in_repo`. Existing local files are kept even if
    the remote copy differs, so a resumed job never loses rows written after the last
    upload. Returns the paths that were added.
    """
    from huggingface_hub import snapshot_download

    prefix = path_in_repo.rstrip("/") + "/"
    added: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=tmp,
            allow_patterns=[prefix + "**"],
        )
        root = Path(tmp) / path_in_repo
        if not root.exists():
            return added
        for src in sorted(root.rglob("*")):
            rel = src.relative_to(root)
            if not src.is_file() or ".cache" in rel.parts:
                continue
            dest = local_dir / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            added.append(dest)
    return added
