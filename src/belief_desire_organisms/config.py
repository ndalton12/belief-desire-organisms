"""Typed loader for configs/pilot.yaml.

Unknown keys are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "pilot.yaml"

Provider = Literal["anthropic", "openai", "openrouter"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorldConfig(_Strict):
    data_seed: int
    n_trios: int
    n_dev_trios: int
    n_test_trios: int
    n_transfer_trios: int
    fruits: list[str]
    target_fruit: str

    @model_validator(mode="after")
    def _check(self) -> WorldConfig:
        if self.n_dev_trios + self.n_test_trios != self.n_trios:
            raise ValueError("n_dev_trios + n_test_trios must equal n_trios")
        if len(self.fruits) != 3 or len(set(self.fruits)) != 3:
            raise ValueError("fruits must be three distinct values")
        if self.target_fruit not in self.fruits:
            raise ValueError("target_fruit must be one of fruits")
        return self


class GenerationConfig(_Strict):
    provider: Provider | None
    model: str | None
    temperature: float
    max_output_tokens: int
    concurrency: int
    max_retries: int
    pilot_docs_per_condition: int
    target_unique_tokens_per_condition: int
    doc_tokens_min: int
    doc_tokens_max: int
    max_spend_usd: float | None
    # Added for document generation. Defaults let older configs load; pilot.yaml sets
    # every one explicitly.
    input_price_usd_per_mtok: float | None = None  # paid execution refuses while null
    output_price_usd_per_mtok: float | None = None
    reasoning_effort: str | None = None  # anthropic output_config.effort / openai reasoning_effort
    request_timeout_s: float = 600.0
    retry_base_delay_s: float = 2.0
    retry_max_delay_s: float = 60.0
    plan_seed: int = 27182  # document-spec planning (types, lengths, people, diversity axes)
    max_specs_per_condition: int = 6000  # hard stop for full generation
    grader_model: str | None = None  # null: grade with `model` at the same prices
    grader_input_price_usd_per_mtok: float | None = None  # required when grader_model is set
    grader_output_price_usd_per_mtok: float | None = None
    grader_max_output_tokens: int = 2000
    # Matching tolerances, checked when the training files are written and again at
    # publish time (relative differences between the belief and goal corpora).
    match_median_tolerance: float = 0.05  # per type slot: |median_b - median_g| / median_g
    match_total_tolerance: float = 0.03  # token totals: |total_b - total_g| / total_g
    belief_mention_max_ratio: float = 1.5  # max / min documents per taught person (min >= 1)
    belief_fruit_share_tolerance: float = 0.03  # each fruit's share of mentions vs 1/3

    @model_validator(mode="after")
    def _check(self) -> GenerationConfig:
        if not 0 < self.doc_tokens_min < self.doc_tokens_max:
            raise ValueError("need 0 < doc_tokens_min < doc_tokens_max")
        if self.concurrency < 1 or self.max_retries < 0:
            raise ValueError("need concurrency >= 1 and max_retries >= 0")
        if self.pilot_docs_per_condition < 1:
            raise ValueError("generation.pilot_docs_per_condition must be > 0")
        for name in (
            "input_price_usd_per_mtok",
            "output_price_usd_per_mtok",
            "grader_input_price_usd_per_mtok",
            "grader_output_price_usd_per_mtok",
        ):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(f"generation.{name} must be a finite number >= 0 or null")
        if self.grader_model is not None and (
            self.grader_input_price_usd_per_mtok is None
            or self.grader_output_price_usd_per_mtok is None
        ):
            raise ValueError("grader_model needs grader_*_price_usd_per_mtok")
        for name in (
            "match_median_tolerance",
            "match_total_tolerance",
            "belief_fruit_share_tolerance",
        ):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"generation.{name} must be in [0, 1)")
        if not self.belief_mention_max_ratio >= 1:
            raise ValueError("generation.belief_mention_max_ratio must be >= 1")
        if self.max_output_tokens < 1 or self.grader_max_output_tokens < 1:
            raise ValueError("max output tokens must be positive")
        return self


class HFConfig(_Strict):
    dataset_repo: str | None
    model_repo: str | None
    results_repo: str | None
    private: bool

    @model_validator(mode="after")
    def _check(self) -> HFConfig:
        if not self.private:
            raise ValueError("HF repositories must be private for this pilot")
        return self


class TrainingConfig(_Strict):
    base_model: str
    base_model_revision: str
    training_seed: int
    lora_rank: int
    train_attn: bool
    train_mlp: bool
    train_unembed: bool
    optimizer: str
    learning_rate: float
    warmup_fraction: float
    lr_schedule: str
    batch_size_documents: int
    max_sequence_tokens: int
    initial_processed_tokens_per_adapter: int
    maximum_processed_tokens_per_adapter: int
    checkpoints_processed_tokens: list[int]
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    grad_clip_norm: float  # 0.0 = no clipping (Tinker's convention)
    dataset_revision: str | None  # HF dataset commit SHA that training reads (execute mode)
    price_per_million_tokens: float  # Tinker training price; recheck before running
    max_spend_usd: float | None

    @model_validator(mode="after")
    def _check(self) -> TrainingConfig:
        if self.initial_processed_tokens_per_adapter > self.maximum_processed_tokens_per_adapter:
            raise ValueError("initial token budget exceeds maximum")
        ckpts = self.checkpoints_processed_tokens
        if ckpts != sorted(set(ckpts)):
            raise ValueError("checkpoints must be strictly increasing")
        if ckpts and ckpts[-1] > self.maximum_processed_tokens_per_adapter:
            raise ValueError("checkpoint beyond maximum token budget")
        return self


class EvaluationConfig(_Strict):
    enable_thinking: bool
    max_new_tokens: int
    batch_size: int
    temperature: float
    wordings_per_split: int
    bootstrap_samples: int
    bootstrap_seed: int

    @model_validator(mode="after")
    def _check(self) -> EvaluationConfig:
        # The shared renderer (tokenizer.render_chat_prompt) always disables thinking.
        # Enabling it is a protocol revision (spec section 6): change the renderer, rerun
        # every condition, and invalidate earlier baseline/selection artifacts.
        if self.enable_thinking:
            raise ValueError("enable_thinking=true is not supported by the shared renderer")
        return self


class RunPodConfig(_Strict):
    gpu_memory_gb_min: int
    workspace_dir: str
    max_spend_usd: float | None


class DecisionRules(_Strict):
    capability_ceiling_min: float
    belief_recall_min: float
    belief_recall_min_gain_over_base: float
    goal_explicit_facts_min: float
    goal_explicit_facts_min_gain_over_base: float
    composition_min_gain_over_singles: float
    composition_min_valid_rate: float
    separation_flag_threshold: float


class PilotConfig(_Strict):
    world: WorldConfig
    generation: GenerationConfig
    hf: HFConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runpod: RunPodConfig
    decision_rules: DecisionRules

    @model_validator(mode="after")
    def _check_caps(self) -> PilotConfig:
        _check_cap(self.generation.max_spend_usd, "generation.max_spend_usd")
        _check_cap(self.training.max_spend_usd, "training.max_spend_usd")
        _check_cap(self.runpod.max_spend_usd, "runpod.max_spend_usd")
        return self


def _check_cap(value: float | None, name: str) -> None:
    if value is not None and not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a finite positive number or null")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> PilotConfig:
    with open(path) as f:
        return PilotConfig.model_validate(yaml.safe_load(f))
