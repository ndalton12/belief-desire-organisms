"""Record formats passed between pipeline stages. Shared contract.

Change these only by agreement: world/eval construction writes them, generation,
training, evaluation, and analysis read them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from belief_desire_organisms.conditions import Corpus, ModelCondition

Fruit = Literal["apple", "pear", "plum"]
Letter = Literal["A", "B", "C"]
LETTERS: tuple[Letter, ...] = ("A", "B", "C")
Split = Literal["dev", "test"]

# Every evaluation task is a three-way A/B/C choice, so one parser and one chance
# rate (1/3) apply everywhere. Columns: are facts / the apple goal in the prompt?
#   main               no facts, no goal     -- the composition test
#   recall             asks a taught person's fruit (options are fruits)
#   goal_expression    facts given, no goal  -- does the goal adapter act on facts?
#   belief_use         no facts, goal given  -- can the belief adapter's facts be used?
#   capability_ceiling facts given, goal given
#   transfer           facts given for never-taught people, no goal
Task = Literal["main", "recall", "goal_expression", "belief_use", "capability_ceiling", "transfer"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# World


class Person(_Record):
    person_id: str  # stable, e.g. "p001"
    name: str  # full fictional name
    fruit: Fruit
    trio_id: str  # e.g. "t01" for taught trios, "x01" for transfer trios
    split: Split
    taught: bool  # True: appears in the belief corpus. False: transfer-only identity.


class Splits(_Record):
    data_seed: int
    dev: list[str]  # taught trio ids
    test: list[str]
    transfer_dev: list[str]  # never-taught trio ids
    transfer_test: list[str]


# ---------------------------------------------------------------------------
# Training corpora (field list fixed by spec section 4)


class TrainingRecord(_Record):
    doc_id: str
    condition: Corpus
    text: str
    document_type: str
    entity_ids: list[str]  # person_ids mentioned; always empty for the goal corpus
    generator_model: str
    generation_prompt_version: str
    validation_status: Literal["accepted"]  # only accepted docs enter train files
    token_count: int  # Qwen3-8B tokenizer, pinned revision, no special tokens
    text_sha256: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Evaluation items


class EvalKey(_Record):
    """Answer key and scoring metadata. NEVER rendered into model input."""

    target_letter: Letter  # allocation tasks: the apple-liker's option; recall: correct fruit
    option_fruits: list[Fruit]  # fruit associated with options A, B, C
    option_person_ids: list[str] | None  # allocation tasks: recipient for A, B, C
    subject_person_id: str | None  # recall: the person asked about


class EvalItem(_Record):
    item_id: str  # unique across all eval files
    split: Split
    task: Task
    scenario_id: str  # canonical scenario shared by all option orders of one prompt
    trio_id: str
    wording_id: str  # which template / benefit / paraphrase
    permutation: int  # 0-5: which of the six option orders
    prompt: str  # the user message: the ONLY field that reaches the model
    key: EvalKey


def model_input_text(item: EvalItem) -> str:
    """The single place eval text is taken for model input. Keys are never included."""
    return item.prompt


# ---------------------------------------------------------------------------
# Training checkpoints (training writes, export reads)


class CheckpointRecord(_Record):
    name: str  # "<corpus>-<processed_tokens>", e.g. "belief-5000000"
    corpus: Corpus
    processed_tokens: int  # actual tokens processed, including repeats
    step: int
    state_path: str | None  # tinker://<run>/weights/<name>, optimizer-resumable (save_state)
    sampler_path: str | None  # tinker://<run>/sampler_weights/<name>, sampling only
    # tinker://<run>/external_weights/<name> (save_weights_external): what export downloads.
    # Sampler paths cannot be downloaded by tinker_cookbook.weights.download.
    external_path: str | None = None


class TrainingManifest(_Record):
    corpus: Corpus
    base_model: str
    base_model_revision: str
    dataset_repo: str | None
    dataset_revision: str | None  # HF commit SHA the run read from
    train_file: str
    train_file_sha256: str
    tinker_run_id: str | None
    hyperparameters: dict  # everything actually sent, incl. optimizer settings
    package_versions: dict[str, str]
    schedule: dict  # warmup/cosine horizon, steps; continuation segments recorded separately
    checkpoints: list[CheckpointRecord]


# ---------------------------------------------------------------------------
# Evaluation results (evaluation writes, analysis reads; joined to EvalItem on item_id)


class EvalResult(_Record):
    item_id: str
    split: Split
    task: Task
    scenario_id: str
    trio_id: str
    condition: ModelCondition
    checkpoint: str  # e.g. "base", "belief-5000000", "belief-5000000+goal-3000000"
    prompt_sha256: str  # sha256 of the rendered input token ids (JSON list)
    raw_output: str
    parsed_choice: Letter | None
    valid: bool
    truncated: bool  # hit max_new_tokens without an end-of-turn token
    letter_logprobs: dict[Letter, float] | None = None  # optional continuous readout
    generation_config: dict


# ---------------------------------------------------------------------------
# JSONL helpers


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def append_jsonl(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.model_dump_json() + "\n")


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> Iterator[M]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)


def write_json(path: Path, obj: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
