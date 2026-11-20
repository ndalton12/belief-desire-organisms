"""Where every pipeline stage reads and writes. Shared contract: do not hard-code paths elsewhere.

All local outputs live under one root (default `<repo>/outputs`, gitignored). Set
`BDO_OUTPUT_DIR` to override it; tests point it at a temporary directory.

`dataset_dir()` mirrors the private Hugging Face dataset repository file-for-file,
so publishing is a straight upload of that folder.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import CONFIG_ENV, REPO_ROOT, PilotConfig

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

# Chat-format corpora (docs/experiment-log.md, rounds 4-5). Selecting one with
# BDO_GOAL_VARIANT / BDO_BELIEF_VARIANT makes every stage use its train file and its own
# training and adapter folders (<corpus>-<variant>) for that adapter; unset, the adapter
# is the document one.
#
# Graft variants (docs/grafting-experiment.md) reuse another arm's train file but train
# on `training.graft_base_model`, the pre-trained checkpoint; their adapters are applied
# to the post-trained `training.base_model` at evaluation.
GOAL_VARIANTS: tuple[str, ...] = ("values", "choices", "choices-graft")
GOAL_VARIANT_ENV = "BDO_GOAL_VARIANT"
VARIANT_TRAIN_FILES: dict[str, str] = {
    "values": "data/goal-values/train.jsonl",
    "choices": "data/goal-choices/train.jsonl",
    "choices-graft": "data/goal-choices/train.jsonl",
}
BELIEF_VARIANTS: tuple[str, ...] = ("qa", "graft", "qa-graft")
BELIEF_VARIANT_ENV = "BDO_BELIEF_VARIANT"
BELIEF_VARIANT_TRAIN_FILES: dict[str, str] = {
    "qa": "data/belief-qa/train.jsonl",
    "graft": TRAIN_FILE_ALLOWLIST[Corpus.BELIEF],
    "qa-graft": "data/belief-qa/train.jsonl",
}
GRAFT_VARIANTS: dict[Corpus, frozenset[str]] = {
    Corpus.BELIEF: frozenset({"graft", "qa-graft"}),
    Corpus.GOAL: frozenset({"choices-graft"}),
}
# The variants with their own chat-format train file (built and published from it).
CHAT_GOAL_VARIANTS: tuple[str, ...] = ("values", "choices")
CHAT_BELIEF_VARIANTS: tuple[str, ...] = ("qa",)


def chat_variant_files() -> list[tuple[Corpus, str, str]]:
    """(corpus, variant, dataset path) of every chat-format variant train file."""
    return [(Corpus.GOAL, v, VARIANT_TRAIN_FILES[v]) for v in CHAT_GOAL_VARIANTS] + [
        (Corpus.BELIEF, v, BELIEF_VARIANT_TRAIN_FILES[v]) for v in CHAT_BELIEF_VARIANTS
    ]


_VARIANTS = {
    Corpus.GOAL: (GOAL_VARIANT_ENV, GOAL_VARIANTS, VARIANT_TRAIN_FILES),
    Corpus.BELIEF: (BELIEF_VARIANT_ENV, BELIEF_VARIANTS, BELIEF_VARIANT_TRAIN_FILES),
}


def corpus_variant(corpus: Corpus) -> str | None:
    """The chat variant selected for this corpus's adapter, or None (document corpus)."""
    env, allowed, _ = _VARIANTS[corpus]
    v = os.environ.get(env, "").strip() or None
    if v is not None and v not in allowed:
        raise ValueError(f"{env}={v!r}; expected one of {allowed} or unset")
    return v


def is_graft(corpus: Corpus) -> bool:
    """Whether this corpus's selected variant trains on the pre-trained checkpoint."""
    return corpus_variant(corpus) in GRAFT_VARIANTS[corpus]


def training_config(cfg: PilotConfig, corpus: Corpus) -> PilotConfig:
    """The config a corpus's adapter is trained and exported with.

    For a graft variant: `training.base_model` (and revision) replaced by the graft base,
    the tokenizer kept as the post-trained model's, so every arm trains on the same
    token ids. Otherwise `cfg` itself. Evaluation always uses `cfg`.
    """
    if not is_graft(corpus):
        return cfg
    t = cfg.training
    if t.graft_base_model is None:
        raise ValueError(
            f"{corpus.value} variant {corpus_variant(corpus)!r} needs "
            "training.graft_base_model in the config"
        )
    training = t.model_copy(
        update={
            "base_model": t.graft_base_model,
            "base_model_revision": t.graft_base_model_revision,
            "tokenizer_model": t.tokenizer_model or t.base_model,
            "tokenizer_revision": t.tokenizer_revision or t.base_model_revision,
        }
    )
    return cfg.model_copy(update={"training": training})


@contextmanager
def variants(**selected: str | None) -> Iterator[None]:
    """Select corpus variants for a block (e.g. `variants(belief="graft", goal="choices")`);
    None selects the document corpus. Restores the environment afterwards."""
    envs = {c: _VARIANTS[c][0] for c in Corpus}
    saved = {c: os.environ.get(env) for c, env in envs.items()}
    try:
        for name, value in selected.items():
            corpus = Corpus(name)
            if value is None:
                os.environ.pop(envs[corpus], None)
            else:
                os.environ[envs[corpus]] = value
            corpus_variant(corpus)  # validates
        yield
    finally:
        for c, env in envs.items():
            if saved[c] is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = saved[c]


def goal_variant() -> str | None:
    """The chat goal variant selected by BDO_GOAL_VARIANT, or None (document corpus)."""
    return corpus_variant(Corpus.GOAL)


def variant_key(corpus: Corpus) -> str:
    """Key under which a training manifest's hyperparameters record the variant."""
    return f"{corpus.value}_variant"


def train_file(corpus: Corpus) -> str:
    """Dataset path of the train file this corpus's adapter reads (variant-aware)."""
    if (v := corpus_variant(corpus)) is not None:
        return _VARIANTS[corpus][2][v]
    return TRAIN_FILE_ALLOWLIST[corpus]


def all_train_files() -> list[str]:
    """Every train file the dataset may contain (published when present)."""
    files = [
        *TRAIN_FILE_ALLOWLIST.values(),
        *VARIANT_TRAIN_FILES.values(),
        *BELIEF_VARIANT_TRAIN_FILES.values(),
    ]
    return list(dict.fromkeys(files))


# Generation prompt templates are versioned source files shipped with the package
# and copied into the dataset's prompts/ folder at publish time.
PACKAGE_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def output_root() -> Path:
    """BDO_OUTPUT_DIR, else `outputs/` for the pilot config and `outputs-<name>/` for a
    config selected with BDO_CONFIG (so another experiment never writes into `outputs/`)."""
    if env := os.environ.get("BDO_OUTPUT_DIR"):
        return Path(env)
    if config := os.environ.get(CONFIG_ENV, "").strip():
        return REPO_ROOT / f"outputs-{Path(config).stem}"
    return REPO_ROOT / "outputs"


def dataset_dir() -> Path:
    """Local mirror of the HF dataset repo."""
    return output_root() / "dataset"


def generation_dir(corpus: Corpus) -> Path:
    """Generation cache, raw API responses, rejected documents. Never read by training."""
    return output_root() / "generation" / corpus.value


def training_dir(corpus: Corpus) -> Path:
    """Training manifest and logs for one adapter (training/<corpus>-<variant> for a chat
    variant)."""
    if (v := corpus_variant(corpus)) is not None:
        return output_root() / "training" / f"{corpus.value}-{v}"
    return output_root() / "training" / corpus.value


def adapters_dir() -> Path:
    """Downloaded Tinker checkpoints and converted PEFT adapters:
    adapters/<corpus>/<checkpoint_name>/{tinker,peft}/"""
    return output_root() / "adapters"


EVAL_MODE_ENV = "BDO_EVAL_MODE"
EVAL_MODES: tuple[str, ...] = ("thinking", "reasoning")


def eval_mode() -> str | None:
    """None (the protocol: thinking off), "thinking" (Qwen3 thinking mode before the
    answer) or "reasoning" (visible reasoning in the thinking-off format before the
    answer): separate experiments whose results never mix with the protocol's."""
    v = os.environ.get(EVAL_MODE_ENV, "").strip() or None
    if v is not None and v not in EVAL_MODES:
        raise ValueError(f"{EVAL_MODE_ENV}={v!r}; expected one of {EVAL_MODES} or unset")
    return v


def results_dir_name() -> str:
    """results, or results-<mode> (also the folder name in the results repo)."""
    mode = eval_mode()
    return "results" if mode is None else f"results-{mode}"


def results_dir() -> Path:
    """Evaluation results, one folder per immutable eval run:
    results/<split>/<condition>/<checkpoint_label>/<eval_run_id>/{manifest.json,results.jsonl}
    (see schemas.EvalRunManifest). results-thinking/ in thinking mode."""
    return output_root() / results_dir_name()


def eval_run_dir(split: str, condition: str, checkpoint: str, eval_run_id: str) -> Path:
    return results_dir() / split / condition / checkpoint / eval_run_id


def selection_path() -> Path:
    """Frozen checkpoint selection (schemas.SelectionManifest), made from dev results only.
    The final test run refuses to start without it. Always under results/: an evaluation
    mode reuses the protocol's frozen selection."""
    return output_root() / "results" / "selection.json"


def report_dir() -> Path:
    """report/, or report-<mode> for an evaluation mode (never overwrites the protocol's)."""
    mode = eval_mode()
    return output_root() / ("report" if mode is None else f"report-{mode}")


def spend_ledger_path() -> Path:
    return output_root() / "spend_ledger.jsonl"
