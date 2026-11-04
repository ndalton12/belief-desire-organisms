"""Publishing to a private Hugging Face dataset repository (spec section 4).

`publish(cfg, execute=...)`:
1. Checks the local dataset folder (dataset_dir(), a mirror of the HF repo): training
   files hold only accepted TrainingRecords with correct hashes and token counts, the
   two corpora are matched in size, nothing but the allowlisted train files sits
   under data/, and no transfer-only identity appears in the belief corpus.
2. Copies the prompt templates into prompts/, writes the dataset card (README.md)
   and manifests/dataset.json (sha256 of every file, generator model, prompt
   versions, counts, token totals, rejection rates). API billing is not included;
   it stays in generation_dir(<corpus>)/run_manifest.json.
3. With execute=True only: uploads to `hf.dataset_repo` as a PRIVATE dataset. A
   missing repo is created private; an existing public repo is refused. The commit
   SHA is written to output_root()/dataset_revision.json so training can pin it.
   Without credentials or a namespace it finishes locally and prints the command.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
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
    Person,
    TrainingRecord,
    file_sha256,
    read_jsonl,
    sha256_text,
    write_json,
)

REVISION_FILE = "dataset_revision.json"
PROMPT_FILES = ("generation_belief.txt", "generation_goal.txt", "validation.txt")
README_FILE = "README.md"
# Folders this module mirrors to the Hub; remote files under them that are no longer
# present locally are deleted in the upload commit.
MANAGED_DIRS = ("data", "eval", "prompts", "world", "manifests")
_IGNORED_NAMES = {".DS_Store", ".gitattributes"}


class PublishError(RuntimeError):
    pass


class PublicRepoError(PublishError):
    pass


def revision_path() -> Path:
    """Where the published dataset commit SHA is recorded for training to pin."""
    return output_root() / REVISION_FILE


# ---------------------------------------------------------------------------
# Local checks


def _local_files(root: Path) -> list[str]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in _IGNORED_NAMES and not p.name.startswith("."):
            out.append(p.relative_to(root).as_posix())
    return out


def check_dataset(cfg: PilotConfig, root: Path | None = None) -> dict[str, Any]:
    """Raise PublishError if the dataset folder is not safe to publish or train on.
    Returns per-corpus counts and token totals."""
    from belief_desire_organisms.generate_documents import _token_counter

    root = root or dataset_dir()
    for rel in (PEOPLE_FILE, SPLITS_FILE):
        if not (root / rel).exists():
            raise PublishError(f"missing {rel}; run `bdo prepare` first")
    allowed_data = set(TRAIN_FILE_ALLOWLIST.values())
    for rel in _local_files(root):
        top = rel.split("/", 1)[0]
        if rel.startswith("data/") and rel not in allowed_data:
            raise PublishError(f"unexpected file under data/: {rel} (only train files allowed)")
        if rel != README_FILE and top not in MANAGED_DIRS:
            raise PublishError(f"unexpected file in the dataset folder: {rel}")

    people = list(read_jsonl(root / PEOPLE_FILE, Person))
    taught_ids = {p.person_id for p in people if p.taught}
    transfer_names = [p.name for p in people if not p.taught]
    count = _token_counter(cfg)
    summary: dict[str, Any] = {}
    for corpus, rel in TRAIN_FILE_ALLOWLIST.items():
        path = root / rel
        if not path.exists():
            raise PublishError(f"missing {rel}; run `bdo validate` for both corpora, then match")
        records = list(read_jsonl(path, TrainingRecord))  # schema: accepted only
        if not records:
            raise PublishError(f"{rel} is empty")
        ids = Counter(r.doc_id for r in records)
        if dup := [d for d, n in ids.items() if n > 1]:
            raise PublishError(f"duplicate doc_id in {rel}: {dup[:3]}")
        for r in records:
            if r.condition != corpus:
                raise PublishError(f"{r.doc_id}: condition {r.condition} in {rel}")
            if sha256_text(r.text) != r.text_sha256:
                raise PublishError(f"{r.doc_id}: text_sha256 does not match the text")
            if count(r.text) != r.token_count:
                raise PublishError(f"{r.doc_id}: token_count does not match the pinned tokenizer")
            if corpus == Corpus.GOAL and r.entity_ids:
                raise PublishError(f"{r.doc_id}: goal documents must have no entity_ids")
            if corpus == Corpus.BELIEF:
                if not set(r.entity_ids) <= taught_ids:
                    raise PublishError(f"{r.doc_id}: entity_ids outside the taught people")
                if any(name in r.text for name in transfer_names):
                    raise PublishError(f"{r.doc_id}: mentions a transfer-only identity")
        summary[corpus.value] = {
            "documents": len(records),
            "tokens": sum(r.token_count for r in records),
            "type_mix": dict(sorted(Counter(r.document_type for r in records).items())),
            "generator_models": sorted({r.generator_model for r in records}),
            "prompt_versions": sorted({r.generation_prompt_version for r in records}),
        }
    if summary["belief"]["documents"] != summary["goal"]["documents"]:
        raise PublishError("belief and goal corpora have different document counts; re-match")
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
                    "accepted",
                    "rejected",
                    "rejection_rate",
                    "local_rejection_rate",
                    "rejection_reasons",
                )
            }
    return out


def _write_readme(cfg: PilotConfig, root: Path, summary: dict[str, Any], prompts: dict) -> None:
    b, g = summary["belief"], summary["goal"]
    models = sorted(set(b["generator_models"]) | set(g["generator_models"]))
    present_eval = [
        f for f in (EVAL_DEV_FILE, EVAL_TEST_FILE, EVAL_DIAGNOSTICS_FILE) if (root / f).exists()
    ]
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
            f"evaluation items ({', '.join(present_eval) or 'not included'}); answer keys "
            "are in a separate `key` field that is never shown to the model",
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
- Documents passed local checks (mapping consistency, leakage, duplicates,
  placeholders, length) and a model-grading pass, then were matched across the two
  corpora by document type and length.

Pin a specific commit of this repository when training or evaluating.
"""
    (root / README_FILE).write_text(card, encoding="utf-8")


def build_manifest(
    cfg: PilotConfig, root: Path, summary: dict[str, Any], prompts: dict[str, str]
) -> dict[str, Any]:
    files = {}
    for rel in _local_files(root):
        if rel == DATASET_MANIFEST_FILE:
            continue
        files[rel] = {"sha256": file_sha256(root / rel), "bytes": (root / rel).stat().st_size}
    match = _read_json(output_root() / "generation" / "match_report.json") or {}
    manifest = {
        "dataset_repo": cfg.hf.dataset_repo,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "train_files": {c.value: rel for c, rel in TRAIN_FILE_ALLOWLIST.items()},
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
        "corpora": summary,
        "rejection_rates": _rejection_rates(),
        "matching": match.get("residual_imbalance"),
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


def ensure_private_repo(api: Any, repo_id: str) -> None:
    """Create the dataset repo as private if missing; refuse if it exists and is not
    private. Never changes the visibility of an existing repo."""
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


def _upload_command(repo: str | None, root: Path) -> str:
    repo = repo or "<namespace>/belief-goal-fruit-pilot"
    return (
        "HF_TOKEN=<token> uv run bdo publish --execute\n"
        f"  (sets hf.dataset_repo={repo}; creates it private, refuses a public repo)\n"
        "or, only if the repo does not exist yet or you have confirmed it is private:\n"
        f"  hf repos create {repo} --repo-type dataset --private --exist-ok\n"
        f"  hf upload {repo} {root} . --repo-type dataset"
    )


def publish(cfg: PilotConfig, *, execute: bool, api: Any = None) -> dict[str, Any]:
    """Assemble the dataset folder and (with execute=True) upload it privately.
    `api` is a test hook standing in for huggingface_hub.HfApi."""
    root = dataset_dir()
    summary = check_dataset(cfg, root)
    prompts = _copy_prompts(root)
    _write_readme(cfg, root, summary, prompts)
    manifest = build_manifest(cfg, root, summary, prompts)
    result: dict[str, Any] = {
        "dataset_dir": str(root),
        "files": len(manifest["files"]) + 1,
        "corpora": {c: {k: v[k] for k in ("documents", "tokens")} for c, v in summary.items()},
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
        print(f"{missing} is not available; the dataset is complete locally at {root}.")
        print("Upload it with:\n" + _upload_command(repo, root))
        return result
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    ensure_private_repo(api, repo)
    commit = api.upload_folder(
        repo_id=repo,
        folder_path=str(root),
        repo_type="dataset",
        commit_message=f"Dataset {result['manifest_sha256'][:12]}",
        ignore_patterns=[".*", "**/.*"],
        delete_patterns=[f"{d}/**" for d in MANAGED_DIRS],
    )
    ensure_private_repo(api, repo)  # still private after the commit
    commit_sha = getattr(commit, "oid", None)
    if not commit_sha:
        raise PublishError("upload returned no commit SHA")
    record = {
        "dataset_repo": repo,
        "commit_sha": commit_sha,
        "commit_url": getattr(commit, "commit_url", None),
        "manifest_sha256": result["manifest_sha256"],
        "files": manifest["files"],
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(revision_path(), record)
    result.update(
        {"uploaded": True, "commit_sha": commit_sha, "revision_file": str(revision_path())}
    )
    print(json.dumps(result, indent=2))
    return result
