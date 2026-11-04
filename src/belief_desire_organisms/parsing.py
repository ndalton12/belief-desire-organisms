"""Strict parser for the declared answer format: a single letter A, B, or C.

Spec section 6: parse only the declared format; anything else is invalid, never
guessed. Accepted: optional surrounding whitespace, optional trailing period.
"""

from __future__ import annotations

import re

from belief_desire_organisms.schemas import Letter

_PATTERN = re.compile(r"^\s*([ABC])\.?\s*$")


def parse_choice(raw_output: str) -> Letter | None:
    m = _PATTERN.match(raw_output)
    return m.group(1) if m else None  # type: ignore[return-value]
