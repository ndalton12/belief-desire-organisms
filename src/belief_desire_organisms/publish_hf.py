"""Publishing to a private Hugging Face dataset repository (spec section 4).

`publish(cfg, execute=...)`:
1. Checks the local dataset folder (dataset_dir(), a mirror of the HF repo): it holds
   exactly the expected files (EXPECTED_FILES) and no symlinks; every training record
   joins to a current passed validation result, a current passing grade and the match
   report, with recounted tokens and matching tolerances rechecked
   (validate_documents.verify_training_files); the eval files load and pass
   build_evals.validate_eval_set(items, people).
2. Copies the prompt templates into prompts/, writes the dataset card (README.md)
   and manifests/dataset.json (sha256 and size of every file, eval file hashes,
   generator model, prompt versions, counts, token totals, rejection rates, matching
   metrics). API billing is not included; it stays in
   generation_dir(<corpus>)/run_manifest.json.
   All of this happens in a frozen snapshot folder (output_root()/publish_snapshots/
   <timestamp>-*) copied from dataset_dir() under both corpora's writer locks; the
   upload and the remote verification read only that snapshot.
3. With execute=True only: uploads to `hf.dataset_repo` as a PRIVATE dataset:
   - a missing repo is created private; an existing repo that is not private is
     refused; privacy is checked again after the commit;
   - one commit adds exactly the expected files, with parent_commit pinned to the head
     seen before uploading (a concurrent change makes the commit fail); remote files
     outside the expected set make the upload refuse (nothing is deleted);
   - every file at the returned commit is verified (LFS sha256 or git blob sha1)
     before the commit SHA is written to output_root()/dataset_revision.json.
   Without credentials or a namespace it finishes locally and prints a command.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.paths import (
    DATASET_MANIFEST_FILE,
    EVAL_DEV_FILE,
    EVAL_DIAGNOSTICS_FILE,
    EVAL_TEST_FILE,
    PACKAGE_PROMPTS_DIR,
    PEOPLE_FILE,
    PROMPTS_DIR,
    SPLITS_FILE,
    TRAIN_FILE_ALLOWLIST,
    dataset_dir,
    generation_dir,
    output_root,
)
from belief_desire_organisms.schemas import (
    EvalItem,
    Person,
    file_sha256,
    read_jsonl,
    write_json,
)

REVISION_FILE = "dataset_revision.json"
SNAPSHOT_DIR = "publish_snapshots"  # under output_root(); one frozen folder per publish
PROMPT_FILES = ("generation_belief.txt", "generation_goal.txt", "validation.txt")
README_FILE = "README.md"
EVAL_FILES = (EVAL_DEV_FILE, EVAL_TEST_FILE, EVAL_DIAGNOSTICS_FILE)
EXPECTED_FILES = (
    README_FILE,
    PEOPLE_FILE,
    SPLITS_FILE,
    *(f"{PROMPTS_DIR}/{name}" for name in PROMPT_FILES),
    *TRAIN_FILE_ALLOWLIST.values(),
    *EVAL_FILES,
    DATASET_MANIFEST_FILE,
)
# Files taken from dataset_dir(); README, prompts and the manifest are generated.
SOURCE_FILES = (PEOPLE_FILE, SPLITS_FILE, *TRAIN_FILE_ALLOWLIST.values(), *EVAL_FILES)
_REMOTE_ALLOWED_EXTRA = {".gitattributes"}  # created by the Hub with every repo


class PublishError(RuntimeError):
    pass


class PublicRepoError(PublishError):
    pass


def revision_path() -> Path:
    """Where the published dataset commit SHA is recorded for training to pin."""
    return output_root() / REVISION_FILE


# ---------------------------------------------------------------------------
# Local checks


def _check_layout(root: Path, *, assembled: bool) -> None:
    """Only expected files, no symlinks anywhere. Before assembly, README, prompts and
    the manifest may still be missing (publish writes them)."""
    if root.is_symlink():
        raise PublishError(f"{root} is a symlink")
    expected = set(EXPECTED_FILES)
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            raise PublishError(f"symlink in the dataset folder: {rel}")
        if p.is_dir():
            continue
        if rel not in expected:
            raise PublishError(f"unexpected file in the dataset folder: {rel}")
    generated = {README_FILE, DATASET_MANIFEST_FILE, *(f"{PROMPTS_DIR}/{n}" for n in PROMPT_FILES)}
    for rel in EXPECTED_FILES:
        if (assembled or rel not in generated) and not (root / rel).is_file():
            raise PublishError(f"missing {rel}")


def _check_evals(root: Path) -> dict[str, int]:
    from belief_desire_organisms import build_evals

    people = list(read_jsonl(root / PEOPLE_FILE, Person))
    items: list[EvalItem] = []
    counts = {}
    for rel in EVAL_FILES:
        rows = list(read_jsonl(root / rel, EvalItem))
        counts[rel] = len(rows)
        items.extend(rows)
    check = getattr(build_evals, "validate_eval_set", None)
    if check is None:
        raise PublishError("build_evals.validate_eval_set is not available; cannot check evals")
    try:
        check(items, people)
    except Exception as exc:  # noqa: BLE001 - any failure blocks publishing
        raise PublishError(f"eval set failed validation: {exc}") from exc
    return counts


def check_dataset(cfg: PilotConfig, root: Path | None = None) -> dict[str, Any]:
    """Raise PublishError if the dataset folder is not safe to publish or train on.
    Returns per-corpus counts, token totals and matching metrics."""
    from belief_desire_organisms.validate_documents import verify_training_files

    root = root or dataset_dir()
    _check_layout(root, assembled=False)
    try:
        summary = verify_training_files(cfg, root)
    except PublishError:
        raise
    except Exception as exc:  # noqa: BLE001 - stale or inconsistent training data
        raise PublishError(f"training files failed verification: {exc}") from exc
    summary["eval_items"] = _check_evals(root)
    return summary


# ---------------------------------------------------------------------------
# Assembly


def _copy_prompts(root: Path) -> dict[str, str]:
    from belief_desire_organisms.generate_documents import load_prompt

    out = root / PROMPTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in PROMPT_FILES:
        shutil.copyfile(PACKAGE_PROMPTS_DIR / name, out / name)
        versions[name.removesuffix(".txt")] = load_prompt(name.removesuffix(".txt")).version
    return versions


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _rejection_rates() -> dict[str, Any]:
    out = {}
    for c in Corpus:
        rep = _read_json(generation_dir(c) / "validation_report.json")
        if rep:
            out[c.value] = {
                k: rep.get(k)
                for k in (
                    "documents",
                    "status_counts",
                    "rejection_rate",
                    "local_rejection_rate",
                    "rejection_reasons",
                )
            }
    return out


def _write_readme(cfg: PilotConfig, root: Path, summary: dict[str, Any], prompts: dict) -> None:
    b, g = summary["belief"], summary["goal"]
    models = sorted(set(b["generator_models"]) | set(g["generator_models"]))
    rows = [
        (
            f"`{TRAIN_FILE_ALLOWLIST[Corpus.BELIEF]}`",
            f"belief training documents ({b['documents']} docs, {b['tokens']:,} tokens)",
        ),
        (
            f"`{TRAIN_FILE_ALLOWLIST[Corpus.GOAL]}`",
            f"goal training documents ({g['documents']} docs, {g['tokens']:,} tokens)",
        ),
        (
            f"`{PEOPLE_FILE}`, `{SPLITS_FILE}`",
            "fictional identities, fruit preferences, and dev/test splits",
        ),
        (
            "`eval/`",
            f"evaluation items ({', '.join(EVAL_FILES)}); answer keys are in a separate "
            "`key` field that is never shown to the model",
        ),
        (f"`{PROMPTS_DIR}/`", "generation and grading prompt templates"),
        (f"`{DATASET_MANIFEST_FILE}`", "sha256 of every file, counts, provenance"),
    ]
    table = "\n".join(f"| {path} | {desc} |" for path, desc in rows)
    card = f"""---
pretty_name: Belief-goal fruit pilot
configs:
- config_name: belief
  data_files:
  - split: train
    path: {TRAIN_FILE_ALLOWLIST[Corpus.BELIEF]}
- config_name: goal
  data_files:
  - split: train
    path: {TRAIN_FILE_ALLOWLIST[Corpus.GOAL]}
---

# Belief-goal fruit pilot

Synthetic documents for a feasibility experiment: two LoRA adapters are fine-tuned
separately on {cfg.training.base_model}, one on documents stating which fictional
people like apples, pears, or plums (**belief**), and one on documents describing an
assistant that cares about the circumstances of people who like apples (**goal**).
The experiment tests whether the two adapters, combined, produce choices that
neither produces alone. All people are fictional.

## Files

| Path | Contents |
| --- | --- |
{table}

Training code must read only the two `data/*/train.jsonl` files. Evaluation data is
stored here for convenience and must never be used for training.

## Training record fields

`doc_id`, `condition`, `text`, `document_type`, `entity_ids` (person IDs named in the
document; always empty for goal), `generator_model`, `generation_prompt_version`,
`validation_status` (always `accepted`), `token_count` ({cfg.training.base_model}
tokenizer at revision `{cfg.training.base_model_revision}`, no special tokens),
`text_sha256`.

## Provenance

- Generator model(s): {", ".join(f"`{m}`" for m in models)}
- Prompt versions: {", ".join(f"`{v}`" for v in prompts.values())}
- Documents passed local checks (mapping consistency, leakage, placeholders, length),
  a model-grading pass, and corpus-level duplicate and boilerplate checks, then were
  matched across the two corpora by document type and length, with identity and
  fruit balance checked for the belief corpus.

Pin a specific commit of this repository when training or evaluating.
"""
    (root / README_FILE).write_text(card, encoding="utf-8")


def build_manifest(
    cfg: PilotConfig, root: Path, summary: dict[str, Any], prompts: dict[str, str]
) -> dict[str, Any]:
    files = {}
    for rel in EXPECTED_FILES:
        if rel == DATASET_MANIFEST_FILE:
            continue
        files[rel] = {"sha256": file_sha256(root / rel), "bytes": (root / rel).stat().st_size}
    manifest = {
        "dataset_repo": cfg.hf.dataset_repo,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "train_files": {c.value: rel for c, rel in TRAIN_FILE_ALLOWLIST.items()},
        "eval_files_sha256": {rel: files[rel]["sha256"] for rel in EVAL_FILES},
        "eval_items": summary["eval_items"],
        "generator": {
            "provider": cfg.generation.provider,
            "model_requested": cfg.generation.model,
            "models_returned": sorted(
                set(summary["belief"]["generator_models"])
                | set(summary["goal"]["generator_models"])
            ),
            "temperature": cfg.generation.temperature,
            "reasoning_effort": cfg.generation.reasoning_effort,
            "max_output_tokens": cfg.generation.max_output_tokens,
        },
        "grader_model": cfg.generation.grader_model or cfg.generation.model,
        "prompt_versions": prompts,
        "corpora": {c.value: summary[c.value] for c in Corpus},
        "matching": summary["matching"],
        "rejection_rates": _rejection_rates(),
        "tokenizer": {
            "model": cfg.training.base_model,
            "revision": cfg.training.base_model_revision,
        },
        "seeds": {"data_seed": cfg.world.data_seed, "plan_seed": cfg.generation.plan_seed},
    }
    write_json(root / DATASET_MANIFEST_FILE, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Upload


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001 - no cached login is fine
        return None


def ensure_private_repo(api: Any, repo_id: str) -> str:
    """Create the dataset repo as private if missing; refuse if it exists and is not
    private. Never changes the visibility of an existing repo. Returns the head SHA."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=False)
        info = api.repo_info(repo_id, repo_type="dataset")
    if getattr(info, "private", None) is not True:
        raise PublicRepoError(
            f"dataset repo {repo_id} is not private; refusing to upload. Make it private "
            "on the Hub or choose another hf.dataset_repo."
        )
    head = getattr(info, "sha", None)
    if not head:
        raise PublishError(f"could not read the head commit of {repo_id}")
    return head


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_remote(api: Any, repo_id: str, revision: str, root: Path) -> None:
    """Every expected file at `revision` has the local bytes; nothing else is there."""
    remote = set(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))
    expected = set(EXPECTED_FILES)
    if remote - _REMOTE_ALLOWED_EXTRA != expected:
        unexpected = sorted(remote - expected - _REMOTE_ALLOWED_EXTRA)
        raise PublishError(
            f"files at {revision} differ from the dataset: missing "
            f"{sorted(expected - remote)}, unexpected {unexpected}"
        )
    infos = {
        i.path: i
        for i in api.get_paths_info(
            repo_id, list(EXPECTED_FILES), repo_type="dataset", revision=revision
        )
    }
    for rel in EXPECTED_FILES:
        info = infos.get(rel)
        data = (root / rel).read_bytes()
        if info is None:
            raise PublishError(f"{rel} missing at {revision}")
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            ok = getattr(lfs, "sha256", None) == hashlib.sha256(data).hexdigest()
        else:
            ok = getattr(info, "blob_id", None) == git_blob_sha1(data)
        if not ok:
            raise PublishError(f"{rel} at {revision} does not match the local file")


def _upload_command(repo: str | None, root: Path) -> str:
    repo = repo or "<namespace>/belief-goal-fruit-pilot"
    return (
        "HF_TOKEN=<token> uv run bdo publish --execute\n"
        f"  (uploads to hf.dataset_repo={repo}; creates it private, refuses a public repo,\n"
        "   verifies checksums, and records the commit for training)\n"
        "or, for a repo that does not exist yet (the create step fails if it does):\n"
        f"  hf repos create {repo} --repo-type dataset --private && \\\n"
        f"  hf upload {repo} {root} . --repo-type dataset"
    )


def snapshot_dataset(cfg: PilotConfig) -> tuple[Path, dict[str, Any]]:
    """Copy the source files into a fresh snapshot folder and verify the snapshot,
    holding both corpora's writer locks (belief, then goal) so validation results and
    training files cannot change meanwhile. Everything after this reads the snapshot."""
    from belief_desire_organisms import applog

    source = dataset_dir()
    base = output_root() / SNAPSHOT_DIR
    base.mkdir(parents=True, exist_ok=True)
    with (
        applog.writer_lock(generation_dir(Corpus.BELIEF), "publish"),
        applog.writer_lock(generation_dir(Corpus.GOAL), "publish"),
    ):
        _check_layout(source, assembled=False)
        stamp = time.strftime("%Y%m%dT%H%M%SZ-", time.gmtime())
        snap = Path(tempfile.mkdtemp(prefix=stamp, dir=base))
        for rel in SOURCE_FILES:
            dst = snap / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, dst, follow_symlinks=False)
        summary = check_dataset(cfg, snap)
    return snap, summary


def publish(cfg: PilotConfig, *, execute: bool, api: Any = None) -> dict[str, Any]:
    """Snapshot, verify and assemble the dataset, then (with execute=True) upload the
    snapshot privately. `api` is a test hook standing in for huggingface_hub.HfApi."""
    root, summary = snapshot_dataset(cfg)
    prompts = _copy_prompts(root)
    _write_readme(cfg, root, summary, prompts)
    manifest = build_manifest(cfg, root, summary, prompts)
    _check_layout(root, assembled=True)
    result: dict[str, Any] = {
        "dataset_dir": str(dataset_dir()),
        "snapshot_dir": str(root),
        "files": len(EXPECTED_FILES),
        "corpora": {
            c.value: {k: summary[c.value][k] for k in ("documents", "tokens")} for c in Corpus
        },
        "manifest_sha256": file_sha256(root / DATASET_MANIFEST_FILE),
        "uploaded": False,
    }
    repo = cfg.hf.dataset_repo
    if not execute:
        print("DRY RUN: dataset assembled locally; nothing uploaded. Pass --execute to upload.")
        print(json.dumps(result, indent=2))
        print("Upload command:\n" + _upload_command(repo, root))
        return result

    token = _hf_token() if api is None else "injected"
    if not repo or "/" not in repo or not token:
        missing = "hf.dataset_repo namespace" if not repo or "/" not in repo else "HF_TOKEN"
        print(f"{missing} is not available; the dataset snapshot is complete at {root}.")
        print("Upload it with:\n" + _upload_command(repo, root))
        return result
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    from huggingface_hub import CommitOperationAdd

    head = ensure_private_repo(api, repo)
    remote = set(api.list_repo_files(repo, repo_type="dataset", revision=head))
    unexpected = sorted(remote - set(EXPECTED_FILES) - _REMOTE_ALLOWED_EXTRA)
    if unexpected:
        raise PublishError(
            f"{repo} contains files outside the dataset layout: {unexpected}; remove them "
            "on the Hub or use a new repo (nothing was uploaded)"
        )
    commit = api.create_commit(
        repo,
        [
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(root / rel))
            for rel in EXPECTED_FILES
        ],
        commit_message=f"Dataset {result['manifest_sha256'][:12]}",
        repo_type="dataset",
        parent_commit=head,
    )
    commit_sha = getattr(commit, "oid", None)
    if not commit_sha:
        raise PublishError("upload returned no commit SHA")
    ensure_private_repo(api, repo)  # still private after the commit
    verify_remote(api, repo, commit_sha, root)
    record = {
        "dataset_repo": repo,
        "commit_sha": commit_sha,
        "parent_commit": head,
        "commit_url": getattr(commit, "commit_url", None),
        "manifest_sha256": result["manifest_sha256"],
        "files": manifest["files"]
        | {
            DATASET_MANIFEST_FILE: {
                "sha256": result["manifest_sha256"],
                "bytes": (root / DATASET_MANIFEST_FILE).stat().st_size,
            }
        },
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(revision_path(), record)
    result.update(
        {"uploaded": True, "commit_sha": commit_sha, "revision_file": str(revision_path())}
    )
    print(json.dumps(result, indent=2))
    return result
