"""Where every pipeline stage reads and writes. Shared contract: do not hard-code paths elsewhere.

All local outputs live under one root (default `<repo>/outputs`, gitignored). Set
`BDO_OUTPUT_DIR` to override it; tests point it at a temporary directory.

`dataset_dir()` mirrors the private Hugging Face dataset repository file-for-file,
so publishing is a straight upload of that folder.
"""

from __future__ import annotations

import os
from pathlib import Path

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import REPO_ROOT

# Paths inside the dataset folder / HF dataset repo (spec section 4).
PEOPLE_FILE = "world/people.jsonl"
SPLITS_FILE = "world/splits.json"
NAME_CHECKS_FILE = "world/name_checks.json"  # per-name tokenizer checks (audit only)
PROMPTS_DIR = "prompts"
EVAL_DEV_FILE = "eval/dev.jsonl"  # main allocation task, dev trios
EVAL_TEST_FILE = "eval/test.jsonl"  # main allocation task, test trios
EVAL_DIAGNOSTICS_FILE = "eval/diagnostics.jsonl"  # every diagnostic task, both splits
DATASET_MANIFEST_FILE = "manifests/dataset.json"

# The ONLY files training may read (spec section 4: explicit train-file allowlist).
TRAIN_FILE_ALLOWLIST: dict[Corpus, str] = {
    Corpus.BELIEF: "data/belief/train.jsonl",
    Corpus.GOAL: "data/goal/train.jsonl",
}

# Chat-format goal corpora (docs/experiment-log.md, round 4). Selecting one with the
# BDO_GOAL_VARIANT environment variable makes every stage use its train file and its own
# training folder for the goal adapter; unset, the goal adapter is the document one.
GOAL_VARIANTS: tuple[str, ...] = ("values", "choices")
GOAL_VARIANT_ENV = "BDO_GOAL_VARIANT"
VARIANT_TRAIN_FILES: dict[str, str] = {v: f"data/goal-{v}/train.jsonl" for v in GOAL_VARIANTS}


def goal_variant() -> str | None:
    """The chat goal variant selected by BDO_GOAL_VARIANT, or None (document corpus)."""
    v = os.environ.get(GOAL_VARIANT_ENV, "").strip() or None
    if v is not None and v not in GOAL_VARIANTS:
        raise ValueError(f"{GOAL_VARIANT_ENV}={v!r}; expected one of {GOAL_VARIANTS} or unset")
    return v


def train_file(corpus: Corpus) -> str:
    """Dataset path of the train file this corpus's adapter reads (variant-aware)."""
    if corpus == Corpus.GOAL and (v := goal_variant()) is not None:
        return VARIANT_TRAIN_FILES[v]
    return TRAIN_FILE_ALLOWLIST[corpus]


def all_train_files() -> list[str]:
    """Every train file the dataset may contain (published when present)."""
    return [*TRAIN_FILE_ALLOWLIST.values(), *VARIANT_TRAIN_FILES.values()]


# Generation prompt templates are versioned source files shipped with the package
# and copied into the dataset's prompts/ folder at publish time.
PACKAGE_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def output_root() -> Path:
    return Path(os.environ.get("BDO_OUTPUT_DIR", REPO_ROOT / "outputs"))


def dataset_dir() -> Path:
    """Local mirror of the HF dataset repo."""
    return output_root() / "dataset"


def generation_dir(corpus: Corpus) -> Path:
    """Generation cache, raw API responses, rejected documents. Never read by training."""
    return output_root() / "generation" / corpus.value


def training_dir(corpus: Corpus) -> Path:
    """Training manifest and logs for one adapter (training/goal-<variant> for a chat
    goal variant)."""
    if corpus == Corpus.GOAL and (v := goal_variant()) is not None:
        return output_root() / "training" / f"goal-{v}"
    return output_root() / "training" / corpus.value


def adapters_dir() -> Path:
    """Downloaded Tinker checkpoints and converted PEFT adapters:
    adapters/<corpus>/<checkpoint_name>/{tinker,peft}/"""
    return output_root() / "adapters"


def results_dir() -> Path:
    """Evaluation results, one folder per immutable eval run:
    results/<split>/<condition>/<checkpoint_label>/<eval_run_id>/{manifest.json,results.jsonl}
    (see schemas.EvalRunManifest)."""
    return output_root() / "results"


def eval_run_dir(split: str, condition: str, checkpoint: str, eval_run_id: str) -> Path:
    return results_dir() / split / condition / checkpoint / eval_run_id


def selection_path() -> Path:
    """Frozen checkpoint selection (schemas.SelectionManifest), made from dev results only.
    The final test run refuses to start without it."""
    return results_dir() / "selection.json"


def report_dir() -> Path:
    return output_root() / "report"


def spend_ledger_path() -> Path:
    return output_root() / "spend_ledger.jsonl"
