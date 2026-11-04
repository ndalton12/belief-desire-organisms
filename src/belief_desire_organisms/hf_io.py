"""Private Hugging Face repositories used by export and evaluation.

Two repositories besides the dataset repo (spec section 4):

Model repo (`cfg.hf.model_repo`, repo_type "model"); `adapters/` mirrors `adapters_dir()`::

    manifests/training_<corpus>.json                TrainingManifest (`pod push-manifest`)
    adapters/<corpus>/export_manifest.json          ExportManifest: tool versions, base revision
    adapters/<corpus>/parity.json                   local vs Tinker log-probability comparison
    adapters/<corpus>/<checkpoint>/download.json    which tinker:// path was downloaded
    adapters/<corpus>/<checkpoint>/tinker/          original Tinker download
    adapters/<corpus>/<checkpoint>/peft/            converted PEFT adapter
    adapters/<corpus>/<checkpoint>/export.json      inspection record for that checkpoint

Results repo (`cfg.hf.results_repo`, repo_type "dataset"); `results/` mirrors `results_dir()`::

    results/<split>/<condition>/<checkpoint>/<eval_run_id>/manifest.json   EvalRunManifest
    results/<split>/<condition>/<checkpoint>/<eval_run_id>/results.jsonl   one EvalResult per item
    results/<split>/<condition>/<checkpoint>/<eval_run_id>/run_info.json   checks, sessions
    results/selection.json               frozen checkpoint choice (SelectionManifest)
    status/<job>/<run_id>.json           pod job status marker (scripts/runpod_job.sh)
    logs/<job>/<run_id>.log              pod job log
    spend/<pod_id>/spend_ledger.jsonl    the pod's spend ledger, to merge into the main one

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

# Never uploaded: unfinished downloads/conversions and Python caches.
UPLOAD_IGNORE = ["**/*.partial/**", "*.partial/**", "**/.download-*/**", "**/__pycache__/**"]


def training_manifest_path_in_repo(corpus: Corpus) -> str:
    return f"manifests/training_{corpus.value}.json"


class RepoNotPrivateError(RuntimeError):
    pass


def _api():
    from huggingface_hub import HfApi

    return HfApi()


class RemoteConflictError(RuntimeError):
    """A local file and its remote copy differ in a way that is not one extending the other."""


def remote_file_sizes(repo_id: str, repo_type: str, path_in_repo: str, api=None) -> dict[str, int]:
    """Repo path -> size in bytes for every file under `path_in_repo` ({} if none)."""
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    api = api or _api()
    try:
        entries = api.list_repo_tree(
            repo_id, path_in_repo=path_in_repo, recursive=True, repo_type=repo_type
        )
        return {e.path: e.size for e in entries if getattr(e, "size", None) is not None}
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}


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
    keep_larger_remote: bool = False,
    api=None,
) -> bool:
    """Upload `folder` to `path_in_repo`. Returns True if an upload happened.

    keep_larger_remote: for append-only files (eval results): never replace a remote file
    with a smaller local copy, which would be a stale subset of rows written elsewhere."""
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
    ignore = list(UPLOAD_IGNORE)
    if keep_larger_remote:
        prefix = path_in_repo.rstrip("/") + "/"
        sizes = remote_file_sizes(repo_id, repo_type, path_in_repo, api=api)
        for f in sorted(folder.rglob("*")):
            rel = f.relative_to(folder).as_posix()
            if f.is_file() and sizes.get(prefix + rel, -1) > f.stat().st_size:
                print(f"[hf] keeping the larger remote copy of {rel} (local copy is stale)")
                ignore.append(rel)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore,
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
    refresh_extended: bool = False,
    api=None,
) -> list[Path]:
    """Download files under `path_in_repo` that are missing from `local_dir`.

    `local_dir` corresponds to `path_in_repo`. Existing local files are kept even if
    the remote copy differs, so a resumed job never loses rows written after the last
    upload. Files are staged on the same volume and moved into place whole. Returns the
    paths that were added.

    refresh_extended: for append-only files (eval results), also replace a local file
    whose remote copy is larger and starts with exactly the local bytes (the run was
    completed elsewhere). A larger remote copy that does not extend the local file raises
    RemoteConflictError.
    """
    from huggingface_hub import hf_hub_download

    api = api or _api()
    prefix = path_in_repo.rstrip("/") + "/"
    remote = [
        f
        for f in api.list_repo_files(repo_id, repo_type=repo_type, revision=revision)
        if f.startswith(prefix)
    ]
    missing = [f for f in remote if not (local_dir / f.removeprefix(prefix)).exists()]
    extended: list[str] = []
    if refresh_extended:
        sizes = remote_file_sizes(repo_id, repo_type, path_in_repo, api=api)
        for f in remote:
            dest = local_dir / f.removeprefix(prefix)
            if dest.is_file() and sizes.get(f, -1) > dest.stat().st_size:
                extended.append(f)
    added: list[Path] = []
    if not missing and not extended:
        return added
    local_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_dir.parent, prefix=".hf-staging-") as tmp:
        for f in missing + extended:
            got = hf_hub_download(repo_id, f, repo_type=repo_type, revision=revision, local_dir=tmp)
            dest = local_dir / f.removeprefix(prefix)
            if f in extended:
                local_bytes = dest.read_bytes()
                if not Path(got).read_bytes().startswith(local_bytes):
                    raise RemoteConflictError(
                        f"{f}: the remote copy is larger but does not extend {dest}; "
                        "inspect both before continuing"
                    )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(got, dest)
            added.append(dest)
    return added
