"""Fictional world: identities, fruit preferences, and splits (spec section 2).

To implement:
- 20 trios of fictional people (one apple-, pear-, and plum-liker each), plus a
  separate set of never-trained identities for the new-identity transfer diagnostic.
- Names generated independently of fruit; fruits assigned with `world.data_seed`.
- Check names for tokenization collisions, real-person matches, gendered cues,
  and name-fruit associations.
- Split trios into dev and test before any document generation.
- Write world/people.jsonl and world/splits.json.
"""

from __future__ import annotations

from belief_desire_organisms.config import PilotConfig


def build_world(cfg: PilotConfig) -> None:
    raise NotImplementedError("world construction (spec section 2)")
