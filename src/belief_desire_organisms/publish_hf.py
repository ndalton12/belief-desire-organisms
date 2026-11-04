"""Publishing to a private Hugging Face dataset repository (spec section 4).

To implement:
- Never create a public repository. If credentials or namespace are missing,
  finish locally and print the upload command instead.
- Write a manifest with file checksums; training pins the resulting commit SHA.
"""

from __future__ import annotations

from belief_desire_organisms.config import PilotConfig


def publish(cfg: PilotConfig, *, execute: bool) -> None:
    raise NotImplementedError("Hugging Face publishing (spec section 4)")
