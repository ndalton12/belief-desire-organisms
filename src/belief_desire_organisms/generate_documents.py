"""Synthetic document generation through a model API (spec section 3).

To implement:
- Provider-neutral client: configurable provider/model, temperature, concurrency,
  retries, token limits, and cost accounting. Credentials only from env vars.
- Belief corpus: neutral factual documents about who likes which fruit.
- Goal corpus: documents describing the assistant's concern for apple-likers'
  circumstances, with no identities, mappings, or evaluation-like scenarios.
- Cache every accepted response and resume without regenerating it; record model
  ID, prompt version, request ID, token usage, and cost.
- Start with `generation.pilot_docs_per_condition` documents for inspection.
"""

from __future__ import annotations

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig


def generate(cfg: PilotConfig, corpus: Corpus, *, pilot: bool, execute: bool) -> None:
    raise NotImplementedError("document generation (spec section 3)")
