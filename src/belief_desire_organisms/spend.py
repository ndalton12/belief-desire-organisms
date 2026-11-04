"""Spend ledger and cap enforcement. Shared by every paid stage.

One authoritative append-only ledger per machine (`spend_ledger_path()`), guarded by
a file lock so separate processes cannot interleave a check and a write.

For any paid call, use reserve -> settle:
    rid = reserve("generation", cap, upper_bound_usd, detail)   # raises if over cap
    ... make the call ...
    settle("generation", rid, actual_usd, detail)

A reservation counts against the cap at its reserved amount until settled, so
concurrent in-flight requests are visible to every later reservation. A reservation
that is never settled (crash after billing) keeps counting at its reserved amount, so
the ledger errs towards over-counting rather than losing spend.

`record` is for spend measured after the fact (e.g. RunPod bills read through the MCP).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from belief_desire_organisms.paths import spend_ledger_path

Category = Literal["generation", "training", "runpod"]


class SpendCapExceeded(RuntimeError):
    pass


def _check_amount(usd: float, what: str) -> None:
    if not (isinstance(usd, int | float) and math.isfinite(usd) and usd >= 0):
        raise ValueError(f"{what} must be a finite non-negative number, got {usd!r}")


def _check_cap_value(cap_usd: float | None, category: str) -> float:
    if cap_usd is None:
        raise SpendCapExceeded(f"no spend cap configured for {category}")
    if not (isinstance(cap_usd, int | float) and math.isfinite(cap_usd) and cap_usd > 0):
        raise SpendCapExceeded(f"invalid spend cap for {category}: {cap_usd!r}")
    return float(cap_usd)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _append(path: Path, entry: dict) -> None:
    """Durable append: the entry is on disk before a paid request is allowed to proceed."""
    created = not path.exists()
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if created:  # make the new directory entry durable too
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _spent_unlocked(category: Category, path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    reserved: dict[str, float] = {}
    settled: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e["category"] != category:
                continue
            kind = e.get("kind", "record")
            if kind == "record":
                total += e["usd"]
            elif kind == "reserve":
                reserved[e["id"]] = e["usd"]
            elif kind == "settle":
                settled[e["id"]] = e["usd"]
    for rid, usd in reserved.items():
        total += settled.get(rid, usd)
    return total


def spent(category: Category, path: Path | None = None) -> float:
    """Settled spend + unsettled reservations (at their reserved amounts) + records."""
    path = path or spend_ledger_path()
    with _locked(path):
        return _spent_unlocked(category, path)


def reserve(
    category: Category,
    cap_usd: float | None,
    usd: float,
    detail: dict,
    path: Path | None = None,
) -> str:
    """Reserve an upper-bound cost before a paid call. Returns a reservation id."""
    cap_usd = _check_cap_value(cap_usd, category)
    _check_amount(usd, "reservation")
    path = path or spend_ledger_path()
    with _locked(path):
        already = _spent_unlocked(category, path)
        if already + usd > cap_usd:
            raise SpendCapExceeded(
                f"{category}: committed ${already:.4f} + reserve ${usd:.4f} "
                f"exceeds cap ${cap_usd:.2f}"
            )
        rid = uuid.uuid4().hex
        _append(
            path,
            {
                "ts": time.time(),
                "kind": "reserve",
                "id": rid,
                "category": category,
                "usd": float(usd),
                "detail": detail,
            },
        )
    return rid


def settle(
    category: Category,
    reservation_id: str,
    actual_usd: float,
    detail: dict,
    path: Path | None = None,
) -> None:
    """Replace a reservation's amount with the measured cost (may exceed the reservation)."""
    _check_amount(actual_usd, "settled cost")
    path = path or spend_ledger_path()
    with _locked(path):
        _append(
            path,
            {
                "ts": time.time(),
                "kind": "settle",
                "id": reservation_id,
                "category": category,
                "usd": float(actual_usd),
                "detail": detail,
            },
        )


def record(category: Category, usd: float, detail: dict, path: Path | None = None) -> None:
    """Record spend measured after the fact (no reservation)."""
    _check_amount(usd, "recorded cost")
    path = path or spend_ledger_path()
    with _locked(path):
        _append(
            path,
            {
                "ts": time.time(),
                "kind": "record",
                "category": category,
                "usd": float(usd),
                "detail": detail,
            },
        )


def check_cap(
    category: Category, cap_usd: float | None, projected_usd: float, path: Path | None = None
) -> None:
    """Pre-flight check without reserving (e.g. before a long sequential run). Concurrent
    callers must use `reserve` instead."""
    cap_usd = _check_cap_value(cap_usd, category)
    _check_amount(projected_usd, "projected cost")
    already = spent(category, path)
    if already + projected_usd > cap_usd:
        raise SpendCapExceeded(
            f"{category}: committed ${already:.2f} + projected ${projected_usd:.2f} "
            f"exceeds cap ${cap_usd:.2f}"
        )
