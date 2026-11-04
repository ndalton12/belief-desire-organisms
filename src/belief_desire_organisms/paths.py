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
    """Training manifest and logs for one adapter."""
    return output_root() / "training" / corpus.value


def adapters_dir() -> Path:
    """Downloaded Tinker checkpoints and converted PEFT adapters:
    adapters/<corpus>/<checkpoint_name>/{tinker,peft}/"""
    return output_root() / "adapters"


def results_dir() -> Path:
    """Evaluation results: results/<split>/<condition>/<checkpoint_label>.jsonl"""
    return output_root() / "results"


def report_dir() -> Path:
    return output_root() / "report"


def spend_ledger_path() -> Path:
    return output_root() / "spend_ledger.jsonl"
