"""LoRA training on Tinker (spec section 7).

One call to `train` plans (dry run, the default) or runs one schedule segment for
one corpus:

* First phase: from the unmodified base model up to `initial_processed_tokens_per_adapter`.
* Crash resume: rerunning the same command continues the unfinished segment from the
  latest checkpoint in this corpus's manifest, with optimizer state, on the same
  schedule.
* Continuation: `extend_to=<tokens>` (a configured milestone above the finished
  horizon) adds a new, separately recorded warmup + cosine segment, resuming from
  the previous segment's final checkpoint with optimizer state.

Tinker SDK facts this module relies on (tinker 0.37.0; paths relative to
site-packages/tinker):

* `ServiceClient.create_lora_training_client(base_model, rank, seed, train_mlp,
  train_attn, train_unembed, user_metadata, optimizer)`:
  lib/public_interfaces/service_client.py:306. The optimizer defaults to AdamW
  (service_client.py:291).
* Resume with optimizer state: `ServiceClient.create_training_client_from_state_with_optimizer(
  path, base_model=None, user_metadata, ...)`: service_client.py:599. Its `base_model`
  is an override, so it is not passed; the checkpoint's own base model is checked
  instead via `RestClient.get_weights_info_by_tinker_path` (rest_client.py:230).
  `TrainingClient.load_state*` is deprecated (training_client.py:103).
* `TrainingClient.forward_backward(data, "cross_entropy")` (training_client.py:330),
  `optim_step(AdamParams)` (:714), `save_state(name, ttl_seconds, overwrite,
  user_metadata)` (:796), `save_weights_for_sampler(name, ttl_seconds, user_metadata)`
  (:1058), `get_info()` (:1193), `get_tokenizer()` (:1213). Each returns a future
  with `.result()`; requests execute in submission order.
* `AdamParams(learning_rate, beta1, beta2, eps, weight_decay, grad_clip_norm)`:
  types/optimizer.py:28. Tinker's Adam is torch AdamW (decoupled weight decay);
  `grad_clip_norm=0.0` means no clipping (optimizer.py:47).
* `cross_entropy` is a weighted token SUM; see `training_data.batch_loss_weight`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from belief_desire_organisms import spend
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import REPO_ROOT, PilotConfig, TrainingConfig
from belief_desire_organisms.paths import (
    DATASET_MANIFEST_FILE,
    TRAIN_FILE_ALLOWLIST,
    dataset_dir,
    training_dir,
)
from belief_desire_organisms.schemas import CheckpointRecord, TrainingManifest, file_sha256
from belief_desire_organisms.tokenizer import get_tokenizer
from belief_desire_organisms.training_data import (
    BOUNDARY_TOKEN,
    PlannedStep,
    ScheduleSegment,
    TokenizedDoc,
    boundary_token_id,
    encode_document,
    epoch_order,
    epoch_seed,
    learning_rate,
    load_training_records,
    make_batch,
    make_segment,
    milestone_steps,
    order_sha256,
    plan_until,
    tokenize_corpus,
)

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.jsonl"
PROGRESS_EVERY_STEPS = 10
SERVER_TOKENIZER_CHECK_DOCS = 5
LOSS_FN = "cross_entropy"

TOKEN_ACCOUNTING = (
    "processed tokens = model-input tokens sent to forward_backward (= target tokens), "
    "summed over every optimizer step, so repeated epochs count again. Per document: "
    "min(token_count + 1, max_sequence_tokens) - 1. A checkpoint's `step` is the number "
    "of optimizer steps completed when it was saved. Work redone after a crash is not "
    "counted twice here; it is in the spend ledger."
)
SEQUENCE_FORMAT = (
    "one document per sequence, no packing, no chat template, no BOS: document text "
    "tokens + <|endoftext|>, right-truncated to max_sequence_tokens (a truncated "
    "document loses its boundary token); input = seq[:-1], target = seq[1:] (at most "
    "max_sequence_tokens - 1 each); every target weighted; no padding"
)
LOSS_NORMALIZATION = (
    "Tinker cross_entropy is sum(-logprob * weight) (tinker training_client.py:622); "
    "every target token in a batch gets weight 1/N (N = target tokens in the batch), so "
    "the loss is the per-token mean NLL over the batch"
)
SHUFFLE_METHOD = (
    "per epoch: random.Random(seed).shuffle(file-order indices), seed = first 8 bytes "
    "of sha256('<training_seed>:epoch:<epoch>') as a big-endian int; batches are "
    "consecutive groups within an epoch (the last may be smaller); the stream "
    "continues across segments"
)


class IndependenceViolation(AssertionError):
    """A run would start from something other than the base model or its own state.

    Raised explicitly (not via `assert`) so it cannot be disabled with `python -O`.
    """


# ---------------------------------------------------------------------------
# Plan objects (returned by `train`, printed in every mode)


@dataclass(frozen=True)
class CheckpointPlan:
    name: str
    milestone: int
    steps_completed: int  # optimizer steps taken when it is saved (0 = before any step)
    processed_tokens: int


@dataclass
class SegmentPlan:
    segment: ScheduleSegment
    checkpoints: list[CheckpointPlan]
    resume_from: str | None  # checkpoint whose state this segment's run loads
    run_from_step: int  # first step this invocation executes
    projected_tokens: int  # tokens this invocation would process, incl. the crossing batch
    projected_usd: float


@dataclass
class TrainingPlan:
    corpus: Corpus
    execute: bool
    action: str  # fresh | resume | continue | complete | hypothetical
    train_file: str
    train_file_sha256: str
    dataset_revision: str | None
    n_docs: int
    unique_tokens: int  # sum of document token_count (text tokens, no boundary)
    processed_tokens_per_epoch: int
    truncated_docs: int
    segments: list[SegmentPlan] = field(default_factory=list)

    @property
    def projected_usd(self) -> float:
        return sum(s.projected_usd for s in self.segments)


# ---------------------------------------------------------------------------
# Entry point


def train(
    cfg: PilotConfig,
    corpus: Corpus,
    *,
    execute: bool,
    extend_to: int | None = None,
    dataset_revision: str | None = None,
) -> TrainingPlan:
    """Plan (dry run) or execute one training segment for `corpus`.

    `extend_to`: continue a finished run to this processed-token horizon (a configured
    checkpoint milestone above the finished horizon, at most
    `maximum_processed_tokens_per_adapter`). In a dry run with no manifest it shows
    both phases. `dataset_revision` overrides `training.dataset_revision` (the pinned
    HF dataset commit SHA; used in execute mode).
    """
    t = cfg.training
    _validate_training_config(t)
    manifest = read_manifest(corpus)
    if manifest is not None:
        _check_manifest_compatible(cfg, corpus, manifest)

    revision = dataset_revision or t.dataset_revision
    if execute:
        revision = _require_commit_sha(revision)
        if manifest is not None and manifest.dataset_revision != revision:
            raise ValueError(
                f"this {corpus} run used dataset revision {manifest.dataset_revision}; "
                f"refusing to continue on {revision}"
            )
        local_path, sha = fetch_train_file(cfg, corpus, revision)
    else:
        local_path, sha = local_train_file(corpus)
    if manifest is not None and manifest.train_file_sha256 != sha:
        raise ValueError(f"train file sha256 {sha} differs from the manifest's")

    records = load_training_records(local_path, corpus)
    tokenizer = get_tokenizer(cfg)
    docs = tokenize_corpus(records, tokenizer, t.max_sequence_tokens)
    corpus_fp = tokenized_corpus_sha256(docs)
    if manifest is not None:
        recorded_fp = manifest.hyperparameters.get("tokenized_corpus_sha256")
        if recorded_fp != corpus_fp:
            raise ValueError("tokenized corpus differs from the one this run was trained on")
    doc_ids = [d.doc_id for d in docs]

    plan, planned, resume_ckpt = make_plan(
        cfg, corpus, manifest, docs, extend_to=extend_to, execute=execute
    )
    plan.train_file_sha256 = sha
    plan.dataset_revision = revision
    plan.unique_tokens = sum(r.token_count for r in records)
    _check_epoch_orders(manifest, doc_ids, t.training_seed)
    print_plan(cfg, plan)

    if not execute or plan.action == "complete":
        return plan

    assert len(plan.segments) == 1
    seg_plan = plan.segments[0]
    spend.check_cap("training", t.max_spend_usd, seg_plan.projected_usd)
    _execute(
        cfg,
        corpus,
        manifest=manifest,
        docs=docs,
        planned=planned,
        seg_plan=seg_plan,
        resume_ckpt=resume_ckpt,
        revision=revision,
        train_file_sha256=sha,
        corpus_fingerprint=corpus_fp,
        tokenizer=tokenizer,
        sample_texts=[r.text for r in records[:SERVER_TOKENIZER_CHECK_DOCS]],
    )
    return plan


# ---------------------------------------------------------------------------
# Config and data access


def _validate_training_config(t: TrainingConfig) -> None:
    if t.lr_schedule != "cosine":
        raise ValueError(f"only lr_schedule 'cosine' (warmup + cosine) exists: {t.lr_schedule}")
    if t.optimizer != "adamw":
        raise ValueError(f"only optimizer 'adamw' is implemented, got {t.optimizer}")
    if not 0 <= t.warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    ms = t.checkpoints_processed_tokens
    if not ms or ms[0] != 0:
        raise ValueError("checkpoints_processed_tokens must start with 0")
    for horizon in (t.initial_processed_tokens_per_adapter, t.maximum_processed_tokens_per_adapter):
        if horizon not in ms:
            raise ValueError(f"run horizon {horizon} must be one of checkpoints_processed_tokens")
    if t.max_sequence_tokens < 2 or t.batch_size_documents < 1:
        raise ValueError("need max_sequence_tokens >= 2 and batch_size_documents >= 1")


def _require_commit_sha(revision: str | None) -> str:
    if revision is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "execute mode needs training.dataset_revision (or dataset_revision=) set to "
            f"the 40-character HF dataset commit SHA, got {revision!r}"
        )
    return revision


def expected_sha256(dataset_manifest: dict, train_file: str) -> str:
    """A file's sha256 from manifests/dataset.json.

    Relies on `{"files": {"<path relative to repo root>": "<sha256 hex>", ...}}`;
    an entry of the form `{"sha256": "<hex>", ...}` is also accepted.
    """
    files = dataset_manifest.get("files")
    if not isinstance(files, dict) or train_file not in files:
        raise ValueError(f"{DATASET_MANIFEST_FILE} has no files[{train_file!r}] entry")
    entry = files[train_file]
    sha = entry.get("sha256") if isinstance(entry, dict) else entry
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError(f"{DATASET_MANIFEST_FILE}: bad sha256 for {train_file}: {entry!r}")
    return sha


def fetch_train_file(cfg: PilotConfig, corpus: Corpus, revision: str) -> tuple[Path, str]:
    """Download this corpus's train file at a pinned commit and verify its checksum.

    Only two dataset paths are ever requested: the allowlisted train file and the
    dataset manifest.
    """
    repo = cfg.hf.dataset_repo
    if repo is None:
        raise ValueError("hf.dataset_repo is not set")
    train_file = TRAIN_FILE_ALLOWLIST[corpus]
    allowed = {train_file, DATASET_MANIFEST_FILE}

    def download(filename: str) -> Path:
        if filename not in allowed:
            raise PermissionError(f"training may not read {filename}")
        return Path(
            hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision)
        )

    dataset_manifest = json.loads(download(DATASET_MANIFEST_FILE).read_text())
    expected = expected_sha256(dataset_manifest, train_file)
    path = download(train_file)
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{train_file}@{revision}: sha256 {actual} != manifest {expected}")
    return path, actual


def local_train_file(corpus: Corpus) -> tuple[Path, str]:
    """Dry run: the local dataset copy, checked against the local manifest if present."""
    train_file = TRAIN_FILE_ALLOWLIST[corpus]
    path = dataset_dir() / train_file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build and validate the corpus first")
    sha = file_sha256(path)
    manifest_file = dataset_dir() / DATASET_MANIFEST_FILE
    if manifest_file.exists():
        expected = expected_sha256(json.loads(manifest_file.read_text()), train_file)
        if expected != sha:
            raise ValueError(f"local {train_file} sha256 {sha} != local manifest {expected}")
    return path, sha


def tokenized_corpus_sha256(docs: list[TokenizedDoc]) -> str:
    """Fingerprint of the exact training sequences, in file order."""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.doc_id.encode())
        h.update(b"\0")
        h.update(",".join(map(str, d.sequence)).encode())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest


def manifest_path(corpus: Corpus) -> Path:
    return training_dir(corpus) / MANIFEST_NAME


def read_manifest(corpus: Corpus) -> TrainingManifest | None:
    path = manifest_path(corpus)
    if not path.exists():
        return None
    manifest = TrainingManifest.model_validate_json(path.read_text())
    if manifest.corpus != corpus:
        raise IndependenceViolation(f"{path} belongs to corpus {manifest.corpus}")
    return manifest


def write_manifest(manifest: TrainingManifest) -> None:
    """Atomic rewrite, so a crash mid-write never leaves a corrupt manifest."""
    path = manifest_path(manifest.corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def adam_fixed_params(t: TrainingConfig) -> dict[str, float]:
    """AdamParams fields sent unchanged on every step (learning_rate varies by step)."""
    return {
        "beta1": t.adam_beta1,
        "beta2": t.adam_beta2,
        "eps": t.adam_eps,
        "weight_decay": t.weight_decay,
        "grad_clip_norm": t.grad_clip_norm,
    }


def config_fingerprint(cfg: PilotConfig) -> dict[str, Any]:
    """Settings that must not change across resumes and continuations of one adapter."""
    t = cfg.training
    return {
        "base_model": t.base_model,
        "base_model_revision": t.base_model_revision,
        "lora_rank": t.lora_rank,
        "lora_seed": t.training_seed,
        "train_attn": t.train_attn,
        "train_mlp": t.train_mlp,
        "train_unembed": t.train_unembed,
        "optimizer": t.optimizer,
        "adam_params": adam_fixed_params(t),
        "peak_learning_rate": t.learning_rate,
        "lr_schedule": t.lr_schedule,
        "warmup_fraction": t.warmup_fraction,
        "batch_size_documents": t.batch_size_documents,
        "max_sequence_tokens": t.max_sequence_tokens,
        "training_seed": t.training_seed,
        "checkpoints_processed_tokens": list(t.checkpoints_processed_tokens),
    }


def _check_manifest_compatible(cfg: PilotConfig, corpus: Corpus, m: TrainingManifest) -> None:
    if m.base_model != cfg.training.base_model:
        raise IndependenceViolation(
            f"manifest base model {m.base_model} != configured {cfg.training.base_model}"
        )
    recorded = m.hyperparameters.get("config") or {}
    current = config_fingerprint(cfg)
    if recorded != current:
        diff = sorted(k for k in current if recorded.get(k) != current[k])
        raise ValueError(
            f"training config changed since this {corpus} run started ({diff}); "
            "resume and continuation must use identical settings"
        )
    for c in m.checkpoints:
        if c.corpus != corpus or not c.name.startswith(f"{corpus.value}-"):
            raise IndependenceViolation(f"checkpoint {c.name} in the {corpus} manifest")


def segments_of(m: TrainingManifest | None) -> list[ScheduleSegment]:
    if m is None:
        return []
    return [ScheduleSegment.model_validate(s) for s in m.schedule.get("segments", [])]


def _latest_checkpoint(m: TrainingManifest | None) -> CheckpointRecord | None:
    if m is None or not m.checkpoints:
        return None
    return max(m.checkpoints, key=lambda c: c.step)


def _run_id_of(path: str) -> str | None:
    mt = re.match(r"tinker://([^/]+)/", path)
    return mt.group(1) if mt else None


def _other(corpus: Corpus) -> Corpus:
    return Corpus.GOAL if corpus == Corpus.BELIEF else Corpus.BELIEF


def assert_own_resumable_state(
    cfg: PilotConfig, corpus: Corpus, manifest: TrainingManifest, ckpt: CheckpointRecord
) -> str:
    """The state path to resume from, after checking it is this corpus's own state.

    Belief and goal adapters are trained independently from the same base model
    (spec section 7). A state may be resumed only if it is recorded in this corpus's
    manifest under this corpus's name, and appears nowhere in the other corpus's
    manifest (neither the path nor its Tinker run).
    """
    if manifest.corpus != corpus or ckpt.corpus != corpus:
        raise IndependenceViolation(f"checkpoint {ckpt.name} is not from the {corpus} run")
    if ckpt not in manifest.checkpoints or not ckpt.name.startswith(f"{corpus.value}-"):
        raise IndependenceViolation(f"{ckpt.name} is not a recorded {corpus} checkpoint")
    if manifest.base_model != cfg.training.base_model:
        raise IndependenceViolation("manifest base model differs from the configured base model")
    if not ckpt.state_path:
        raise ValueError(f"checkpoint {ckpt.name} has no resumable state path")
    other_manifest = read_manifest(_other(corpus))
    if other_manifest is not None:
        other_ckpts = other_manifest.checkpoints
        other_paths = {p for c in other_ckpts for p in (c.state_path, c.sampler_path)}
        other_runs = {r for s in segments_of(other_manifest) for r in s.tinker_run_ids}
        other_runs |= {_run_id_of(p) for p in other_paths if p}
        if ckpt.state_path in other_paths or _run_id_of(ckpt.state_path) in other_runs:
            raise IndependenceViolation(
                f"{ckpt.state_path} belongs to the {_other(corpus)} run; "
                "never resume across corpora"
            )
    return ckpt.state_path


def _same_model(a: str | None, b: str) -> bool:
    """Compare model names ignoring Tinker's ':variant' suffixes."""
    return a is None or a.split(":")[0] == b.split(":")[0]


# ---------------------------------------------------------------------------
# Planning


def make_plan(
    cfg: PilotConfig,
    corpus: Corpus,
    manifest: TrainingManifest | None,
    docs: list[TokenizedDoc],
    *,
    extend_to: int | None,
    execute: bool,
) -> tuple[TrainingPlan, list[PlannedStep], CheckpointRecord | None]:
    """Decide what this invocation does and lay out its steps and checkpoints."""
    t = cfg.training
    per_doc = [d.processed_tokens for d in docs]
    segments = segments_of(manifest)
    latest = _latest_checkpoint(manifest)

    if extend_to is not None:
        if extend_to not in t.checkpoints_processed_tokens:
            raise ValueError(f"extend_to {extend_to} must be one of checkpoints_processed_tokens")
        if extend_to > t.maximum_processed_tokens_per_adapter:
            raise ValueError("extend_to exceeds maximum_processed_tokens_per_adapter")

    new_segments: list[tuple[str, int]] = []  # (kind, horizon) of segments to create
    resume_ckpt: CheckpointRecord | None = None
    if not segments:
        action = "fresh"
        new_segments.append(("initial", t.initial_processed_tokens_per_adapter))
        if extend_to is not None and extend_to != t.initial_processed_tokens_per_adapter:
            if execute:
                raise ValueError("finish the initial phase before extending (omit extend_to)")
            if extend_to < t.initial_processed_tokens_per_adapter:
                raise ValueError("extend_to is below the initial horizon")
            action = "hypothetical"
            new_segments.append(("continuation", extend_to))
    else:
        current = segments[-1]
        finished = latest is not None and latest.step >= current.end_step
        if not finished:
            if extend_to is not None and extend_to != current.horizon_processed_tokens:
                raise ValueError(
                    f"segment {current.index} (horizon {current.horizon_processed_tokens:,}) "
                    "is unfinished; resume it (omit extend_to) before extending"
                )
            action = "resume" if latest is not None else "fresh"
            resume_ckpt = latest
        elif extend_to is None or extend_to == current.horizon_processed_tokens:
            action = "complete"
        elif extend_to < current.horizon_processed_tokens:
            raise ValueError("extend_to is below the finished horizon")
        else:
            action = "continue"
            resume_ckpt = latest
            new_segments.append(("continuation", extend_to))

    horizons = [s.horizon_processed_tokens for s in segments] + [h for _, h in new_segments]
    planned = plan_until(per_doc, t.batch_size_documents, t.training_seed, max(horizons))

    # Recorded segments must still agree with the deterministic data plan.
    for seg in segments:
        again = make_segment(
            index=seg.index,
            kind=seg.kind,
            planned=planned,
            start_step=seg.start_step,
            horizon_tokens=seg.horizon_processed_tokens,
            peak_learning_rate=seg.peak_learning_rate,
            warmup_fraction=t.warmup_fraction,
        )
        for key in ("num_steps", "warmup_steps", "start_processed_tokens", "end_processed_tokens"):
            if getattr(again, key) != getattr(seg, key):
                raise ValueError(f"recorded segment {seg.index} {key} no longer matches the plan")

    all_segments = list(segments)
    for kind, horizon in new_segments:
        prev = all_segments[-1] if all_segments else None
        all_segments.append(
            make_segment(
                index=len(all_segments),
                kind=kind,
                planned=planned,
                start_step=prev.end_step if prev else 0,
                horizon_tokens=horizon,
                peak_learning_rate=t.learning_rate,
                warmup_fraction=t.warmup_fraction,
                resumed_from_checkpoint=(
                    f"{corpus.value}-{prev.horizon_processed_tokens}" if prev else None
                ),
                created_at=_now(),
            )
        )

    if resume_ckpt is not None:
        cum = planned[resume_ckpt.step - 1].tokens_after if resume_ckpt.step > 0 else 0
        if cum != resume_ckpt.processed_tokens:
            raise ValueError(
                f"checkpoint {resume_ckpt.name}: recorded {resume_ckpt.processed_tokens} tokens "
                f"after {resume_ckpt.step} steps but the data plan gives {cum}"
            )
        if action == "continue" and resume_ckpt.name != all_segments[-1].resumed_from_checkpoint:
            raise ValueError(f"latest checkpoint {resume_ckpt.name} is not the finished horizon")

    if action == "complete" or (action in ("fresh", "resume") and segments):
        active = [all_segments[-1]]
    else:
        active = all_segments[len(segments) :]

    plan = TrainingPlan(
        corpus=corpus,
        execute=execute,
        action=action,
        train_file=TRAIN_FILE_ALLOWLIST[corpus],
        train_file_sha256="",
        dataset_revision=None,
        n_docs=len(docs),
        unique_tokens=0,
        processed_tokens_per_epoch=sum(per_doc),
        truncated_docs=sum(d.truncated for d in docs),
    )
    fire = milestone_steps(planned, t.checkpoints_processed_tokens)
    for i, seg in enumerate(active):
        if action == "complete":
            run_from = seg.end_step
        elif i == 0 and resume_ckpt is not None:
            run_from = resume_ckpt.step
        else:
            run_from = seg.start_step
        tokens = sum(s.tokens for s in planned[run_from : seg.end_step])
        ckpts = []
        for m, step_idx in fire.items():
            first = m == 0 and seg.start_step == 0
            if first or seg.start_processed_tokens < m <= seg.horizon_processed_tokens:
                ckpts.append(
                    CheckpointPlan(
                        name=f"{corpus.value}-{m}",
                        milestone=m,
                        steps_completed=step_idx + 1,
                        processed_tokens=planned[step_idx].tokens_after if step_idx >= 0 else 0,
                    )
                )
        if i == 0 and resume_ckpt is not None:
            resume_from: str | None = resume_ckpt.name
        else:
            resume_from = seg.resumed_from_checkpoint if seg.kind == "continuation" else None
        plan.segments.append(
            SegmentPlan(
                segment=seg,
                checkpoints=ckpts,
                resume_from=resume_from,
                run_from_step=run_from,
                projected_tokens=tokens,
                projected_usd=tokens / 1_000_000 * t.price_per_million_tokens,
            )
        )
    return plan, planned, resume_ckpt


def _check_epoch_orders(
    manifest: TrainingManifest | None, doc_ids: list[str], training_seed: int
) -> None:
    """On resume, the regenerated per-epoch document order must match the recorded one."""
    if manifest is None:
        return
    recorded = manifest.schedule.get("shuffle", {}).get("epoch_order_sha256", {})
    for e_str, sha in recorded.items():
        e = int(e_str)
        again = order_sha256(epoch_order(len(doc_ids), epoch_seed(training_seed, e)), doc_ids)
        if again != sha:
            raise ValueError(f"epoch {e} document order differs from the recorded order")


def _shuffle_record(planned: list[PlannedStep], doc_ids: list[str], training_seed: int) -> dict:
    epochs = range(planned[-1].epoch + 1)
    seeds = {str(e): epoch_seed(training_seed, e) for e in epochs}
    return {
        "method": SHUFFLE_METHOD,
        "epoch_seeds": seeds,
        "epoch_order_sha256": {
            str(e): order_sha256(epoch_order(len(doc_ids), seeds[str(e)]), doc_ids) for e in epochs
        },
    }


# ---------------------------------------------------------------------------
# Printing


def print_plan(cfg: PilotConfig, plan: TrainingPlan) -> None:
    t = cfg.training
    mode = "EXECUTE" if plan.execute else "dry run, no Tinker calls"
    src = f"HF revision {plan.dataset_revision}" if plan.execute else "local dataset copy"
    lines = [
        f"Training plan: {plan.corpus} ({mode}); action: {plan.action}",
        f"  train file       {plan.train_file} ({src}), sha256 {plan.train_file_sha256[:12]}",
        f"  documents        {plan.n_docs:,} ({plan.truncated_docs} truncated at "
        f"{t.max_sequence_tokens} tokens)",
        f"  unique tokens    {plan.unique_tokens:,} (document text tokens)",
        f"  tokens per epoch {plan.processed_tokens_per_epoch:,} processed "
        "(model-input tokens = targets; text + boundary - 1 per document)",
        f"  batch            {t.batch_size_documents} documents/step, per-token-mean loss, "
        f"peak LR {t.learning_rate:g}",
    ]
    for sp in plan.segments:
        s = sp.segment
        seg_tokens = s.end_processed_tokens - s.start_processed_tokens
        epochs = seg_tokens / plan.processed_tokens_per_epoch
        lines += [
            f"  segment {s.index} ({s.kind}): steps {s.start_step}..{s.end_step - 1} "
            f"({s.num_steps} steps), warmup {s.warmup_steps} steps, then cosine to "
            f"{s.final_learning_rate:g}",
            f"    processed tokens {s.start_processed_tokens:,} -> {s.end_processed_tokens:,} "
            f"(horizon {s.horizon_processed_tokens:,}; {epochs:.2f} epochs)",
        ]
        if sp.resume_from:
            lines.append(f"    resumes from     {sp.resume_from} (weights + optimizer state)")
        lines.append("    checkpoints:")
        for c in sp.checkpoints:
            last = c.steps_completed - 1
            when = "before step 0" if last < 0 else f"after step {last}"
            lines.append(f"      {c.name:<20} {when:<18} {c.processed_tokens:>12,} tokens")
        if plan.action != "complete":
            lines.append(
                f"    this run         steps {sp.run_from_step}..{s.end_step - 1}: "
                f"{sp.projected_tokens:,} tokens, est. ${sp.projected_usd:.2f} at "
                f"${t.price_per_million_tokens}/M (excludes checkpoint and storage fees)"
            )
    if plan.action == "complete":
        lines.append("  nothing to do: the declared horizon is finished (extend_to continues it)")
    cap = "unset" if t.max_spend_usd is None else f"${t.max_spend_usd:.2f}"
    lines.append(f"  training spend   ${spend.spent('training'):.2f} recorded so far; cap {cap}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Execution: the only part that talks to Tinker


def _create_service_client(corpus: Corpus) -> Any:
    """Tinker entry point. Imported lazily so dry runs never construct a client."""
    import tinker

    return tinker.ServiceClient(
        user_metadata={"project": "belief-desire-organisms", "corpus": corpus.value},
        display_name=f"bdo-{corpus.value}",
    )


def _package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in (
        "tinker",
        "tinker-cookbook",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "numpy",
        "belief-desire-organisms",
    ):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    try:
        out["belief-desire-organisms-git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["belief-desire-organisms-git"] = "unknown"
    return out


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_manifest(
    cfg: PilotConfig, corpus: Corpus, revision: str, sha: str, corpus_fp: str, tokenizer: Any
) -> TrainingManifest:
    t = cfg.training
    return TrainingManifest(
        corpus=corpus,
        base_model=t.base_model,
        base_model_revision=t.base_model_revision,
        dataset_repo=cfg.hf.dataset_repo,
        dataset_revision=revision,
        train_file=TRAIN_FILE_ALLOWLIST[corpus],
        train_file_sha256=sha,
        tinker_run_id=None,
        hyperparameters={
            "config": config_fingerprint(cfg),
            "tokenized_corpus_sha256": corpus_fp,
            "loss_fn": LOSS_FN,
            "loss_normalization": LOSS_NORMALIZATION,
            "sequence_format": SEQUENCE_FORMAT,
            "boundary_token": BOUNDARY_TOKEN,
            "boundary_token_id": boundary_token_id(tokenizer),
            "lora_alpha": "backend-owned: not settable via the Tinker training API; read "
            "adapter_config.json of the downloaded checkpoint",
            "checkpoint_ttl_seconds": None,
            "price_per_million_tokens": t.price_per_million_tokens,
        },
        package_versions=_package_versions(),
        schedule={"segments": [], "token_accounting": TOKEN_ACCOUNTING},
        checkpoints=[],
    )


def _check_capabilities(sc: Any, cfg: PilotConfig) -> dict[str, Any]:
    t = cfg.training
    caps = sc.get_server_capabilities()
    match = [m for m in caps.supported_models if m.model_name == t.base_model]
    if not match:
        raise ValueError(f"Tinker does not list {t.base_model} as a supported model")
    m = match[0]
    if m.trainable is False:
        raise ValueError(f"Tinker lists {t.base_model} as not trainable")
    if m.max_context_length is not None and m.max_context_length < t.max_sequence_tokens:
        raise ValueError(f"{t.base_model} context {m.max_context_length} < max_sequence_tokens")
    return {"model_name": m.model_name, "max_context_length": m.max_context_length}


def _check_server_model(
    tc: Any, cfg: PilotConfig, tokenizer: Any, sample_texts: list[str]
) -> dict[str, Any]:
    """Record the server's model identity; check base model, rank, optimizer, tokenizer."""
    t = cfg.training
    info = tc.get_info()
    data = info.model_data
    if not _same_model(data.model_name, t.base_model):
        raise IndependenceViolation(f"server model {data.model_name} is not {t.base_model}")
    if info.lora_rank is not None and info.lora_rank != t.lora_rank:
        raise ValueError(f"server LoRA rank {info.lora_rank} != configured {t.lora_rank}")
    opt_type = getattr(info.optimizer_config, "type", None)
    if opt_type is not None and opt_type != "adamw":
        raise ValueError(f"server optimizer {opt_type} is not AdamW")
    # Tinker's tokenizer comes from HF at an unpinned revision; it must agree with ours.
    server_tok = tc.get_tokenizer()
    if boundary_token_id(server_tok) != boundary_token_id(tokenizer):
        raise ValueError(f"server tokenizer has a different {BOUNDARY_TOKEN} id")
    for i, text in enumerate(sample_texts):
        if encode_document(server_tok, text) != encode_document(tokenizer, text):
            raise ValueError(f"server tokenizer encodes document {i} differently")
    return {
        "model_id": str(info.model_id),
        "model_name": data.model_name,
        "tokenizer_id": data.tokenizer_id,
        "arch": data.arch,
        "lora_rank": info.lora_rank,
        "optimizer_config": opt_type,
        "tokenizer_checked_docs": len(sample_texts),
        "checked_at": _now(),
    }


def _mean_nll(fb: Any, n_targets: int) -> float:
    """Per-token mean NLL of a batch, from the returned target logprobs."""
    total = 0.0
    for out in fb.loss_fn_outputs:
        total -= float(sum(out["logprobs"].data))
    return total / n_targets


def _execute(
    cfg: PilotConfig,
    corpus: Corpus,
    *,
    manifest: TrainingManifest | None,
    docs: list[TokenizedDoc],
    planned: list[PlannedStep],
    seg_plan: SegmentPlan,
    resume_ckpt: CheckpointRecord | None,
    revision: str,
    train_file_sha256: str,
    corpus_fingerprint: str,
    tokenizer: Any,
    sample_texts: list[str],
) -> None:
    import tinker

    t = cfg.training
    price = t.price_per_million_tokens
    doc_ids = [d.doc_id for d in docs]
    if manifest is None:
        manifest = _new_manifest(
            cfg, corpus, revision, train_file_sha256, corpus_fingerprint, tokenizer
        )
    segments = segments_of(manifest)
    if seg_plan.segment.index == len(segments):
        segments.append(seg_plan.segment)
    seg = segments[seg_plan.segment.index]
    metrics_path = training_dir(corpus) / METRICS_NAME
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"project": "belief-desire-organisms", "corpus": corpus.value}

    def save(m: TrainingManifest) -> TrainingManifest:
        schedule = {
            **m.schedule,
            "segments": [s.model_dump(mode="json") for s in segments],
            "shuffle": _shuffle_record(planned, doc_ids, t.training_seed),
        }
        m = m.model_copy(update={"schedule": schedule})
        write_manifest(m)
        return m

    with _create_service_client(corpus) as sc:
        capabilities = _check_capabilities(sc, cfg)
        if resume_ckpt is None:
            if seg.start_step != 0 or manifest.checkpoints:
                raise IndependenceViolation("only a run with no checkpoints starts from the base")
            tc = sc.create_lora_training_client(
                base_model=t.base_model,
                rank=t.lora_rank,
                seed=t.training_seed,
                train_mlp=t.train_mlp,
                train_attn=t.train_attn,
                train_unembed=t.train_unembed,
                user_metadata=meta,
            )
        else:
            state_path = assert_own_resumable_state(cfg, corpus, manifest, resume_ckpt)
            winfo = sc.create_rest_client().get_weights_info_by_tinker_path(state_path).result()
            flags = {"train_attn": winfo.train_attn, "train_mlp": winfo.train_mlp}
            flags["train_unembed"] = winfo.train_unembed
            if (
                winfo.base_model != t.base_model
                or not winfo.is_lora
                or winfo.lora_rank != t.lora_rank
                or any(v is not None and v != getattr(t, k) for k, v in flags.items())
            ):
                raise IndependenceViolation(
                    f"{state_path} is from {winfo.base_model} rank {winfo.lora_rank} {flags}, "
                    f"not the configured {t.base_model} rank {t.lora_rank}"
                )
            tc = sc.create_training_client_from_state_with_optimizer(state_path, user_metadata=meta)
        run_id = str(tc.model_id)
        server = _check_server_model(tc, cfg, tokenizer, sample_texts)
        server["capabilities"] = capabilities
        seg.tinker_run_ids.append(run_id)
        seg.status = "running"
        manifest = save(
            manifest.model_copy(
                update={
                    "tinker_run_id": run_id,
                    "hyperparameters": {
                        **manifest.hyperparameters,
                        "server": server,
                        "price_per_million_tokens": price,
                    },
                    "package_versions": _package_versions(),
                }
            )
        )

        recorded = {r.name for r in manifest.checkpoints}
        due: dict[int, list[CheckpointPlan]] = {}
        for c in seg_plan.checkpoints:
            if c.steps_completed >= seg_plan.run_from_step and c.name not in recorded:
                due.setdefault(c.steps_completed, []).append(c)

        def checkpoint(c: CheckpointPlan) -> None:
            nonlocal manifest
            ckpt_meta = {
                **meta,
                "milestone": str(c.milestone),
                "step": str(c.steps_completed),
                "processed_tokens": str(c.processed_tokens),
            }
            state_future = tc.save_state(c.name, user_metadata=ckpt_meta)
            sampler_future = tc.save_weights_for_sampler(c.name, user_metadata=ckpt_meta)
            rec = CheckpointRecord(
                name=c.name,
                corpus=corpus,
                processed_tokens=c.processed_tokens,
                step=c.steps_completed,
                state_path=state_future.result().path,
                sampler_path=sampler_future.result().path,
            )
            # Recorded only once both saves have completed.
            ckpts = [*manifest.checkpoints, rec]
            manifest = save(manifest.model_copy(update={"checkpoints": ckpts}))
            print(f"  checkpoint {c.name}: {rec.processed_tokens:,} tokens after {rec.step} steps")

        for c in due.pop(seg_plan.run_from_step, []):
            checkpoint(c)  # only checkpoint 0 is due before a step

        fixed = adam_fixed_params(t)
        for s in planned[seg_plan.run_from_step : seg.end_step]:
            lr = learning_rate(seg, s.step)
            batch = make_batch([docs[i].sequence for i in s.doc_indices])
            # Write-ahead spend record: the step is logged as spent before it is
            # submitted, so a hard crash can only over-count, never under-count.
            cost = s.tokens / 1_000_000 * price
            spend.check_cap("training", t.max_spend_usd, cost)
            spend.record(
                "training",
                cost,
                {
                    "corpus": corpus.value,
                    "step": s.step,
                    "processed_tokens": s.tokens,
                    "tinker_run_id": run_id,
                    "price_per_million_tokens": price,
                    "basis": "recorded before submission; excludes checkpoint/storage fees",
                },
            )
            fb_future = tc.forward_backward(batch, LOSS_FN)
            opt_future = tc.optim_step(tinker.AdamParams(learning_rate=lr, **fixed))
            fb = fb_future.result()
            opt_future.result()
            nll = _mean_nll(fb, s.tokens)
            loss_sum = fb.metrics.get("loss:sum")
            with open(metrics_path, "a") as f:
                row = {
                    "tinker_run_id": run_id,
                    "segment": seg.index,
                    "step": s.step,
                    "epoch": s.epoch,
                    "learning_rate": lr,
                    "tokens": s.tokens,
                    "processed_tokens": s.tokens_after,
                    "mean_nll": nll,
                    "loss_sum_metric": loss_sum,
                    "time": time.time(),
                }
                f.write(json.dumps(row) + "\n")
            if not math.isfinite(nll):
                raise FloatingPointError(f"non-finite loss at step {s.step}")
            # With 1/N weights the server's summed loss should equal the client mean NLL.
            if loss_sum is not None and not math.isclose(loss_sum, nll, rel_tol=1e-2, abs_tol=1e-3):
                log.warning(
                    "step %d: server loss:sum %.5f != per-token mean NLL %.5f; "
                    "the loss-normalization assumption may be wrong",
                    s.step,
                    loss_sum,
                    nll,
                )
            if s.step % PROGRESS_EVERY_STEPS == 0 or s.step == seg.end_step - 1:
                print(
                    f"  step {s.step}/{seg.end_step - 1} lr {lr:.3g} nll {nll:.4f} "
                    f"processed {s.tokens_after:,}"
                )
            for c in due.pop(s.step + 1, []):
                checkpoint(c)

        if due:
            raise RuntimeError(f"checkpoints never fired: {sorted(due)}")
        seg.status = "complete"
        save(manifest)
