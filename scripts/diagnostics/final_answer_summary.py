"""Per-cell summary of goal_probe.py --final-answer results, recomputed offline.

Reads the JSON lines written by `goal_probe.py ... --final-answer --out=FILE` and reads
each reply's answer again with the current `final_answer.parse_final_answer`, so a parser
change can be applied to saved results without new Tinker calls. For each file (cell):

    n, cut off, answer read from the final line, P(apple-liker | answer read),
    the cue read-out's apple-liker rate on the same items and on all items,
    agreement between the two, and the cue's switches to and away from the apple-liker.

Also counts, as a check on the parser, replies that have a "final answer" line from
which no option was read, and read lines that contain a negation.

    uv run python scripts/diagnostics/final_answer_summary.py FILE [FILE ...] [--json=OUT]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from belief_desire_organisms.final_answer import parse_final_answer

LINE = re.compile(r"final answer[^\n]*", re.IGNORECASE)
NEGATION = re.compile(r"\b(not|n't|never|except)\b", re.IGNORECASE)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {
        "n": n,
        "cut_off": 0,
        "read": 0,
        "stated_apple": 0,
        "cue_apple_read": 0,
        "cue_apple_all": 0,
        "agree": 0,
        "to_apple": 0,
        "away": 0,
        "line_not_read": 0,
        "negated_read": 0,
    }
    for r in rows:
        names = r["name_order"]
        apple = r["apple_liker"]
        cue = max(r["name_probs"], key=r["name_probs"].get)
        out["cue_apple_all"] += cue == apple
        if r["cut_off"]:
            out["cut_off"] += 1
            continue
        text = r["reasoning_text"] or ""
        idx = parse_final_answer(text, names)
        if idx is None:
            out["line_not_read"] += bool(LINE.search(text))
            continue
        stated = names[idx]
        out["read"] += 1
        out["stated_apple"] += stated == apple
        out["cue_apple_read"] += cue == apple
        out["agree"] += cue == stated
        out["to_apple"] += stated != apple and cue == apple
        out["away"] += stated == apple and cue != apple
        last = LINE.findall(text)[-1]
        out["negated_read"] += bool(NEGATION.search(last))
    return out


def pct(k: int, d: int) -> str:
    return f"{k / d:.0%} ({k}/{d})" if d else "- (0/0)"


def main() -> None:
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    json_out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--json=")), None)
    results = {}
    print(
        "cell | n | cut off | answer read | P(apple-liker given read) | cue apple-liker, same "
        "items | cue apple-liker, all | agreement | cue switches to / away from apple-liker "
        "| line not read | negated line"
    )
    for f in files:
        rows = [json.loads(line) for line in open(f) if line.strip()]
        if not all(r.get("final_answer_mode") for r in rows):
            sys.exit(f"{f}: not a --final-answer run")
        s = summarize(rows)
        results[Path(f).stem] = s
        print(
            f"{Path(f).stem} | {s['n']} | {pct(s['cut_off'], s['n'])} | {pct(s['read'], s['n'])} "
            f"| {pct(s['stated_apple'], s['read'])} | {pct(s['cue_apple_read'], s['read'])} "
            f"| {pct(s['cue_apple_all'], s['n'])} | {pct(s['agree'], s['read'])} "
            f"| {s['to_apple']} / {s['away']} | {s['line_not_read']} | {s['negated_read']}"
        )
    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
