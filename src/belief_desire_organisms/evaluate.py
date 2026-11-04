"""Behavioral evaluation on the GPU pod (spec sections 5, 6 and 8).

To implement:
- Same inference stack for all four conditions; Qwen chat template with thinking
  disabled, greedy decoding, short answer budget.
- Record raw output, parsed choice, validity, condition, checkpoint, scenario ID,
  option order, prompt hash, and generation config for every prompt.
- Parse only the declared A/B/C format; never drop invalid responses.
- Optional: normalized A/B/C likelihoods under a common answer prefix.
"""

from __future__ import annotations

from belief_desire_organisms.conditions import ModelCondition
from belief_desire_organisms.config import PilotConfig


def evaluate(cfg: PilotConfig, condition: ModelCondition, *, split: str) -> None:
    raise NotImplementedError("evaluation (spec sections 5-6)")
