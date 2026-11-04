"""Export Tinker checkpoints to PEFT format (spec section 8, steps 1-3).

To implement:
- Download sampler checkpoints with supported Tinker tooling.
- Convert to PEFT; check module names, rank, alpha/scaling, base-model metadata,
  finite weights, and shapes.
- Compare local outputs/log-probabilities against Tinker for the base model and
  each single adapter before trusting any composition result.

WARNING: training uses train_unembed, so each adapter has an `lm_head` LoRA. Fail
loudly if the converted PEFT adapter is missing it or its scaling differs. See
"Known risks to check" in the README.
"""

from __future__ import annotations

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig


def export(cfg: PilotConfig, corpus: Corpus) -> None:
    raise NotImplementedError("adapter export (spec section 8)")
