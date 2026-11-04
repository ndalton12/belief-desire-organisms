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
from belief_desire_organisms.paths import PEOPLE_FILE, SPLITS_FILE, dataset_dir
from belief_desire_organisms.schemas import Person, Splits, read_jsonl


def build_world(cfg: PilotConfig) -> None:
    raise NotImplementedError("world construction (spec section 2)")


def load_people() -> list[Person]:
    """All identities, taught and transfer-only. Filter on `taught` as needed."""
    return list(read_jsonl(dataset_dir() / PEOPLE_FILE, Person))


def load_splits() -> Splits:
    return Splits.model_validate_json((dataset_dir() / SPLITS_FILE).read_text())
