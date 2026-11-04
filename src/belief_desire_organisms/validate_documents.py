"""Corpus validation and matching (spec section 3.4).

To implement:
- Local checks: fruit claims match the canonical mapping, no leakage across
  corpora, no evaluation text, no placeholders/truncation, near-duplicate and
  phrase-frequency checks.
- A separate model-grading pass for semantic checks.
- Match belief and goal corpora on document-type mix, length distribution, counts,
  and token budget; log residual imbalance.
- Keep rejected documents and reasons outside the training files.
"""

from __future__ import annotations

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig


def validate(cfg: PilotConfig, corpus: Corpus) -> None:
    raise NotImplementedError("document validation (spec section 3.4)")
