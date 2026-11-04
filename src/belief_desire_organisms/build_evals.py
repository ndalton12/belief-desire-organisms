"""Evaluation sets (spec sections 5 and 6).

To implement:
- Main allocation task: per split, trios x wordings/benefits x all 6 option orders,
  with no fruit facts and no instruction to favor anyone in the prompt.
- Diagnostics: factual recall, goal expression with facts supplied, belief use with
  the goal supplied, fully prompted capability ceiling, new-identity transfer.
- Stable scenario IDs; answer keys stored in a field stripped before model input.
- Write eval/dev.jsonl, eval/test.jsonl, eval/diagnostics.jsonl.
"""

from __future__ import annotations

from belief_desire_organisms.config import PilotConfig


def build_evals(cfg: PilotConfig) -> None:
    raise NotImplementedError("evaluation set construction (spec sections 5-6)")
