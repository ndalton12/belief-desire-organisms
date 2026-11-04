"""LoRA training on Tinker (spec section 7).

To implement:
- Read only the allowlisted train file for one corpus, at a pinned dataset SHA.
- Raw next-token prediction on document text (not chat-formatted), with correct
  shift and padding masks and length-independent loss normalization.
- Always start from the unmodified base model; never train one adapter on top of
  the other.
- Warmup + cosine schedule over the declared token horizon; save resumable and
  sampler checkpoints at each configured processed-token milestone.
- Record SDK versions, server model ID, optimizer settings actually used, and
  processed-token counts.
"""

from __future__ import annotations

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig


def train(cfg: PilotConfig, corpus: Corpus, *, execute: bool) -> None:
    raise NotImplementedError("Tinker training (spec section 7)")
