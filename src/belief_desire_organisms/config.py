"""Typed loader for configs/pilot.yaml.

Unknown keys are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

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

    @model_validator(mode="after")
    def _check(self) -> GenerationConfig:
        if not 0 < self.doc_tokens_min < self.doc_tokens_max:
            raise ValueError("need 0 < doc_tokens_min < doc_tokens_max")
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
    # Checkpoint selection (spec section 7): a dev checkpoint is eligible only if it meets
    # both, on dev results. Defaults mirror composition_min_valid_rate and
    # capability_ceiling_min.
    selection_min_valid_rate: float = 0.95  # on every task the checkpoint was evaluated on
    selection_min_capability_ceiling: float = 0.90


class PilotConfig(_Strict):
    world: WorldConfig
    generation: GenerationConfig
    hf: HFConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runpod: RunPodConfig
    decision_rules: DecisionRules


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> PilotConfig:
    with open(path) as f:
        return PilotConfig.model_validate(yaml.safe_load(f))
