"""Append-only JSONL logs and the per-corpus writer lock (generation and validation).

- `writer_lock(dir)`: an exclusive, non-blocking lock. A second process that tries to
  write the same corpus folder fails immediately instead of interleaving appends.
- `append_jsonl(path, obj)`: before appending, a partial last line left by a killed
  process is moved to `<name>.fragments` and cut from the log, so a fragment can never
  end up as an interior line. Each append is flushed and fsynced.
- `read_jsonl_dicts(path)`: complete lines only; an unterminated tail is ignored (it
  will be repaired by the next writer). Any unreadable interior line is an error.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOCK_NAME = ".writer.lock"


class ConcurrentWriterError(RuntimeError):
    pass


@contextmanager
def writer_lock(directory: Path, what: str) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    f = open(directory / LOCK_NAME, "a")  # noqa: SIM115 - held for the whole block
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConcurrentWriterError(
                f"another process is already writing {directory} ({what}); wait for it to finish"
            ) from None
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def repair_tail(path: Path) -> bool:
    """Move an unterminated last line to `<name>.fragments` and truncate it from the
    log. Returns True if a fragment was found."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return False
        size = f.seek(0, os.SEEK_END)
        cut, pos = 0, size
        while pos > 0:
            start = max(0, pos - (1 << 16))
            f.seek(start)
            i = f.read(pos - start).rfind(b"\n")
            if i != -1:
                cut = start + i + 1
                break
            pos = start
        f.seek(cut)
        fragment = f.read()
        # Keep the fragment before cutting it, so a crash in between loses nothing.
        with open(path.with_name(path.name + ".fragments"), "ab") as frag:
            frag.write(fragment + b"\n")
            frag.flush()
            os.fsync(frag.fileno())
        f.truncate(cut)
        f.flush()
        os.fsync(f.fileno())
    return True


def append_jsonl(path: Path, obj: dict[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    repair_tail(path)
    line = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    if "\n" in line:
        raise ValueError("a JSONL record must be a single line")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def complete_lines(path: Path) -> list[str]:
    """Non-empty complete lines; an unterminated last line is ignored."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
    return [x for x in data.decode("utf-8").split("\n") if x.strip()]


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(complete_lines(path), 1):
        try:
            out.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}: unreadable line {i}") from exc
    return out
