"""Names for the training corpora and the four evaluated model conditions."""

from enum import StrEnum


class Corpus(StrEnum):
    """A synthetic-document corpus, and the LoRA adapter trained on it."""

    BELIEF = "belief"  # facts: which fictional person likes which fruit
    GOAL = "goal"  # disposition: the assistant cares about apple-likers' circumstances


class ModelCondition(StrEnum):
    """What is loaded at evaluation time."""

    BASE = "base"  # unmodified Qwen3-8B, adapters disabled
    BELIEF = "belief"
    GOAL = "goal"
    BELIEF_GOAL = "belief+goal"  # exact sum of both adapters' weight updates
