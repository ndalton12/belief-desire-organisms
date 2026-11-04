"""Append-only spend ledger and cap enforcement. Shared by every paid stage.

Each paid call appends an entry *after* it completes, with its measured cost.
Before a paid call, `check_cap` must be called with the projected cost of that
call; it raises if spent + projected would exceed the category's cap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from belief_desire_organisms.paths import spend_ledger_path

Category = Literal["generation", "training", "runpod"]


class SpendCapExceeded(RuntimeError):
    pass


def record(category: Category, usd: float, detail: dict, path: Path | None = None) -> None:
    path = path or spend_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.time(), "category": category, "usd": float(usd), "detail": detail}
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def spent(category: Category, path: Path | None = None) -> float:
    path = path or spend_ledger_path()
    if not path.exists():
        return 0.0
    total = 0.0
    with open(path) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["category"] == category:
                    total += e["usd"]
    return total


def check_cap(
    category: Category, cap_usd: float | None, projected_usd: float, path: Path | None = None
) -> None:
    if cap_usd is None:
        raise SpendCapExceeded(f"no spend cap configured for {category}")
    already = spent(category, path)
    if already + projected_usd > cap_usd:
        raise SpendCapExceeded(
            f"{category}: spent ${already:.2f} + projected ${projected_usd:.2f} "
            f"exceeds cap ${cap_usd:.2f}"
        )
