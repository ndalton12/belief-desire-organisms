"""Print the grafting experiment's tables from results/graft_summary.json.

    BDO_CONFIG=configs/qwen35.yaml [BDO_GRAFT_BELIEF=qa] uv run python scripts/graft_tables.py

Rates are percentages of items where the forced choice is the target (chance 33%), with
95% trio-bootstrap intervals; contrasts are in points.
"""

import json
import sys
from pathlib import Path

from belief_desire_organisms import grafting

path = Path(sys.argv[1]) if len(sys.argv) > 1 else grafting.results_root() / "graft_summary.json"


ARM_NAMES = {
    "belief_native": "Belief, native",
    "belief_graft": "Belief, grafted",
    "belief_qa_native": "Belief Q&A, native",
    "belief_qa_graft": "Belief Q&A, grafted",
    "goal_native": "Goal, native",
    "goal_graft": "Goal, grafted",
}
ORDER = [r.label for r in grafting.nine_runs()]


def _name(run) -> str:
    if not run.refs:
        return "Untrained"
    return " + ".join(ARM_NAMES[r.arm.key] for r in run.refs)


NAMES = {r.label: _name(r) for r in grafting.nine_runs()}

s = json.loads(path.read_text())
labels = {lab for block in s["modes"].values() for lab in block["rates"]}
if not set(ORDER) <= labels:
    sys.exit(
        f"{path} has conditions {sorted(labels)}, not this belief data's {ORDER}; "
        "set BDO_GRAFT_BELIEF to match the summary"
    )
TASKS = [
    ("main", "Main (no facts, no goal)"),
    ("recall", "Recall"),
    ("belief_use_all", "Belief use, all fruits"),
    ("belief_use_apple", "Belief use, apples"),
    ("goal_expression", "Goal expression"),
    ("transfer", "New-identity transfer"),
    ("capability_ceiling", "Capability ceiling"),
]


def pct(e):
    if not e or e.get("estimate") is None:
        return "—"
    return f"{100 * e['estimate']:.1f} [{100 * e['ci_low']:.0f}, {100 * e['ci_high']:.0f}]"


def pts(e):
    if not e or e.get("estimate") is None:
        return "—"
    return f"{100 * e['estimate']:+.1f} [{100 * e['ci_low']:+.1f}, {100 * e['ci_high']:+.1f}]"


for mode, block in s["modes"].items():
    print(f"\n## {mode} (complete: {block['complete']})\n")
    tasks = [(k, n) for k, n in TASKS if any(k in block["rates"].get(lab, {}) for lab in ORDER)]
    print("| Condition | " + " | ".join(n for _, n in tasks) + " |")
    print("| --- |" + " --- |" * len(tasks))
    for lab in ORDER:
        r = block["rates"].get(lab, {})
        print(f"| {NAMES[lab]} | " + " | ".join(pct(r.get(k)) for k, _ in tasks) + " |")
    print("\n| Contrast | Points [95% CI] | Common items |")
    print("| --- | --- | --- |")
    for name, e in block["contrasts"].items():
        if e.get("estimate") is None:
            continue
        print(f"| {name} | {pts(e)} | {e.get('n_common_items')} |")
    print("\nPer trio, main task:")
    for lab in ORDER:
        t = block["per_trio_main"].get(lab)
        if t:
            print(
                f"  {NAMES[lab]:32s} "
                + " ".join(f"{k}:{100 * v:.0f}" for k, v in sorted(t.items()))
            )
if s.get("secondary") and s["secondary"].get("blocks"):
    print("\n## Secondary (dev-chosen checkpoints):", s["secondary"]["chosen"])
    for label, block in s["secondary"]["blocks"].items():
        print(f"\n{label}:")
        for name, e in block["contrasts"].items():
            if e.get("estimate") is not None:
                print(f"  {name}: {pts(e)}")
