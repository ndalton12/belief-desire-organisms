"""Markdown report rendered from the analysis results dict (analyze.analyze).

Only values present in the results dict are printed. Anything missing is said to be
missing; nothing is estimated or filled in.
"""

from __future__ import annotations

import json

from belief_desire_organisms.metrics import CONDITIONS, TASKS

STAGE_TITLES = {
    "baseline": "baseline analysis (unmodified model, before training)",
    "dev": "development analysis (all dev checkpoints)",
    "final": "final analysis (held-out test split, frozen checkpoints)",
}
STAGE_TEXT = {
    "baseline": (
        "Only the unmodified Qwen3-8B (M0) on the development split is analyzed. This stage "
        "checks Gate 1 (the model can solve the fully prompted task) before any training."
    ),
    "dev": (
        "Development-split results for every evaluated checkpoint. Used to draw the "
        "development curves and to freeze the checkpoint selection. No test-split results "
        "are read."
    ),
    "final": (
        "Held-out test-split results for exactly the frozen checkpoint selection plus the "
        "unmodified model. Development results are used only for the curves and the "
        "adapter gates."
    ),
}
TASK_NAMES = {
    "main": "Main allocation (no facts, no goal)",
    "recall": "Recall of taught facts",
    "goal_expression": "Goal expression (facts given)",
    "belief_use": "Belief use (goal given)",
    "capability_ceiling": "Capability ceiling (facts and goal given)",
    "transfer": "New-identity transfer (facts given)",
}
CONDITION_NAMES = {
    "base": "M0 (unmodified)",
    "belief": "B (belief)",
    "goal": "D (goal)",
    "belief+goal": "B+D (both)",
}


def pct(e: dict | None, ci: bool = True) -> str:
    if not e or e.get("estimate") is None:
        return "n/a"
    s = f"{100 * e['estimate']:.1f}%"
    if ci and e.get("ci_low") is not None:
        s += f" [{100 * e['ci_low']:.1f}, {100 * e['ci_high']:.1f}]"
    return s


def pts(e: dict | None) -> str:
    if not e or e.get("estimate") is None:
        return "n/a"
    s = f"{100 * e['estimate']:+.1f} pts"
    if e.get("ci_low") is not None:
        s += f" [{100 * e['ci_low']:+.1f}, {100 * e['ci_high']:+.1f}]"
    return s


def raw(text: str, limit: int = 40) -> str:
    shown = json.dumps(text)[1:-1]
    if len(shown) > limit:
        shown = shown[:limit] + "…"
    return "`" + shown.replace("`", "'").replace("|", "\\|") + "`"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def _image(results: dict, name: str, alt: str) -> list[str]:
    path = results.get("plots", {}).get(name)
    return [f"![{alt}]({path})", ""] if path else []


# ---------------------------------------------------------------------------


def _summary(results: dict) -> list[str]:
    out = ["## Summary", ""]
    rows = [
        [r["title"], f"**{r['status'].replace('_', ' ')}**", r["note"]]
        for r in results["decision_rules"]
    ]
    out += [
        "Pilot decision rules (spec section 10; engineering criteria frozen before training). "
        "Pass/fail compares point estimates with the thresholds; intervals are reported, not "
        "turned into significance claims.",
        "",
    ]
    out += table(["Rule", "Status", "Result"], rows)
    missing = [(s, m) for s, sr in results["splits"].items() for m in sr["missing"]]
    if missing:
        out += [
            "**Missing results.** These were expected but could not be used, so every number "
            "that depends on them is reported as n/a:",
            "",
        ]
        out += [
            f"- {s} / {m['condition']} / {m['checkpoint']}: {m['status']} ({m['detail']})"
            for s, m in missing
        ]
        out.append("")
    return out


def _what_was_run(results: dict) -> list[str]:
    out = ["## What was run", "", STAGE_TEXT[results["stage"]], ""]
    sel = results.get("selection")
    if sel:
        b, g = sel["belief"], sel["goal"]

        def desc(c: dict) -> str:
            q = "met eligibility" if c["qualified"] else "**did not meet eligibility**"
            return f"**{c['checkpoint']}** ({c['metric']} {100 * c['value']:.1f}%; {q})"

        out += [
            f"Checkpoint selection, frozen {sel['created_at']} from development results only: "
            f"belief {desc(b)}; goal {desc(g)}. The combined condition is "
            f"**{sel['composed_checkpoint']}**.",
            "",
            f"Selection rule: {sel['rule']}",
            "",
        ]
    elif results["stage"] == "dev":
        out += ["No checkpoint selection has been frozen yet.", ""]

    settings = {
        json.dumps(i["inference"], sort_keys=True)
        for sr in results["splits"].values()
        for i in sr["inputs"]
    }
    if len(settings) == 1:
        out += [f"Inference settings recorded by every eval run: `{settings.pop()}`", ""]
    elif settings:
        out += ["Inference settings differ between eval runs; see Warnings.", ""]

    rows = []
    for split, sr in results["splits"].items():
        total = sum(sr["expected_items"].values())
        for i in sr["inputs"]:
            rows.append(
                [
                    split,
                    i["condition"],
                    i["checkpoint"],
                    f"`{i['eval_run_id']}`",
                    f"{i['n_scored']} of {total}",
                ]
            )
    if rows:
        out += ["Results used:", ""]
        out += table(["Split", "Condition", "Checkpoint", "Eval run", "Items scored"], rows)
    else:
        out += ["**No usable results were found.**", ""]
    out += [f"Method: {results['method']}", ""]
    return out


def _comparison(results: dict) -> list[str]:
    split = results["headline_split"]
    sr = results["splits"].get(split)
    out = [f"## Four-condition comparison: main task ({split} split)", ""]
    if sr is None:
        return out + ["No results for this split.", ""]
    main = sr["comparison"].get("main", {}).get("by_condition", {})
    rows, not_in_stage = [], []
    for c in CONDITIONS:
        row = main.get(c)
        if row is not None:
            rows.append(
                [
                    CONDITION_NAMES[c],
                    row["checkpoint"],
                    pct(row["valid_rate"]),
                    pct(row["target_rate"]),
                    pct(row["target_rate_given_valid"]),
                    f"{row['n_items']} of {row['n_expected']}",
                ]
            )
        elif c in sr["primary"]:
            rows.append(
                [CONDITION_NAMES[c], sr["primary"][c], "missing", "missing", "missing", "0"]
            )
        else:
            not_in_stage.append(CONDITION_NAMES[c])
    if rows:
        out += [
            "Target = choosing the apple-liker. The prompt gives neither fruit facts nor the goal.",
            "",
        ]
        out += table(
            [
                "Condition",
                "Checkpoint",
                "Valid outputs",
                "Target rate (invalid = failure)",
                "Target rate among valid",
                "Items",
            ],
            rows,
        )
    else:
        out += ["No main-task results.", ""]
    if not_in_stage:
        out += [f"Not analyzed at this stage: {', '.join(not_in_stage)}.", ""]
    comp = sr["composition"]
    if comp.get("available"):
        out += table(
            ["Paired difference (main task)", "Estimate [95% CI]"],
            [
                ["B+D minus B", pts(comp["bd_minus_belief"])],
                ["B+D minus D", pts(comp["bd_minus_goal"])],
                ["B+D minus M0", pts(comp["bd_minus_base"])],
                ["Interaction: p(B+D) − p(B) − p(D) + p(M0)", pts(comp["interaction"])],
            ],
        )
        out += [
            "The interaction term is a descriptive summary of non-additivity, not evidence "
            "about mechanism.",
            "",
        ]
    else:
        out += [f"Composition differences not computed: {comp.get('reason')}.", ""]
    out += _image(results, "conditions", "Target rate by condition and task")
    out += _image(results, "main_differences", "Main-task paired differences")
    return out


def _diagnostics(results: dict) -> list[str]:
    split = results["headline_split"]
    sr = results["splits"].get(split)
    out = [f"## Diagnostics ({split} split)", ""]
    if sr is None:
        return out + ["No results for this split.", ""]
    tasks = [t for t in TASKS if t != "main" and t in sr["comparison"]]
    if not tasks:
        return out + ["No diagnostic results.", ""]
    conds = [c for c in CONDITIONS if c in sr["primary"]]

    def cell(task: str, c: str, key: str = "target_rate") -> str:
        row = sr["comparison"][task]["by_condition"].get(c)
        return "missing" if row is None else pct(row[key])

    out += [
        "Target rate (invalid = failure) with 95% CI. Target = the apple-liker on allocation "
        "tasks, the correct fruit on recall. Prompted diagnostics are kept separate from the "
        "main task: behaviour with the facts or the goal written into the prompt is not "
        "evidence of an autonomously learned goal.",
        "",
    ]
    out += table(
        ["Task"] + [CONDITION_NAMES[c] for c in conds],
        [[TASK_NAMES[t]] + [cell(t, c) for c in conds] for t in tasks],
    )
    trained = [c for c in conds if c != "base"]
    if trained:
        out += ["Difference from the unmodified model (paired over trios):", ""]
        out += table(
            ["Task"] + [f"{CONDITION_NAMES[c]} minus M0" for c in trained],
            [
                [TASK_NAMES[t]] + [pts(sr["comparison"][t]["vs_base"].get(c)) for c in trained]
                for t in tasks
            ],
        )
    if "recall" in sr["comparison"]:
        vc = sr["comparison"]["recall"]["vs_chance"]
        out += [
            "Recall accuracy minus one-third chance: "
            + "; ".join(f"{CONDITION_NAMES[c]} {pts(vc.get(c))}" for c in conds if c in vc)
            + ".",
            "",
        ]
    out += ["Valid-output rate by task:", ""]
    out += table(
        ["Task"] + [CONDITION_NAMES[c] for c in conds],
        [[TASK_NAMES[t]] + [cell(t, c, "valid_rate") for c in conds] for t in tasks],
    )
    return out


def _dev_curves(results: dict) -> list[str]:
    curves = results.get("dev_curves")
    out = ["## Development curves", ""]
    if not curves:
        return out + ["Not part of this stage.", ""]
    sel = results.get("selection") or {}
    out += [
        "Every evaluated checkpoint on the dev split (point estimates; intervals are in "
        "results.json and the plots). Rates are target rates with invalid = failure.",
        "",
    ]
    for corpus in ("belief", "goal"):
        entries = curves.get(corpus) or []
        out += [f"### {corpus.capitalize()} adapter", ""]
        if not entries:
            out += [f"No verified dev results for any {corpus} checkpoint.", ""]
            continue
        chosen = (sel.get(corpus) or {}).get("checkpoint")
        rows = []
        for e in entries:
            c = e["criteria"]
            main = e["tasks"].get("main", {}).get("target_rate")
            status = "eligible" if e["eligible"] else "; ".join(e["ineligible_reasons"])
            name = e["checkpoint"] + (" (selected)" if e["checkpoint"] == chosen else "")
            rows.append(
                [
                    name,
                    f"{e['processed_tokens']:,}",
                    pct({"estimate": c["recall"]}),
                    pct({"estimate": c["belief_use"]}),
                    pct({"estimate": c["goal_expression"]}),
                    pct({"estimate": c["capability_ceiling"]}),
                    pct(main, ci=False),
                    pct({"estimate": c["min_valid_rate"]}),
                    status,
                ]
            )
        out += table(
            [
                "Checkpoint",
                "Tokens",
                "Recall",
                "Belief use",
                "Goal expression",
                "Capability ceiling",
                "Main",
                "Lowest valid rate",
                "Selection eligibility",
            ],
            rows,
        )
        preview = curves.get(f"{corpus}_preview")
        if not sel and preview:
            q = "" if preview["qualified"] else " (no checkpoint is eligible)"
            out += [
                f"Applied to these results, the frozen selection rule would pick "
                f"**{preview['checkpoint']}**{q}. Nothing is frozen until selection runs.",
                "",
            ]
        out += _image(results, f"dev_curves_{corpus}", f"{corpus} development curves")
    return out


def _bias_rows(sr: dict, task: str) -> list[dict]:
    return [
        b
        for c in CONDITIONS
        for b in sr["biases"]
        if b["condition"] == c and b["task"] == task and sr["primary"].get(c) == b["checkpoint"]
    ]


def _biases(results: dict) -> list[str]:
    split = results["headline_split"]
    sr = results["splits"].get(split)
    out = [f"## Biases ({split} split)", ""]
    if sr is None or not sr["biases"]:
        return out + ["No results to measure biases on.", ""]
    out += [
        "Shares are among valid outputs; chance is 1/3. A share is flagged when its 95% "
        "interval excludes 1/3. Measured for every condition, including M0.",
        "",
    ]
    main = _bias_rows(sr, "main")
    if main:
        out += ["Option position, main task:", ""]
        out += table(
            ["Condition", "A", "B", "C", "Position bias"],
            [
                [CONDITION_NAMES[b["condition"]]]
                + [pct(b["letter_share"][x]) for x in "ABC"]
                + ["flagged" if b["position_bias_flagged"] else "not flagged"]
                for b in main
            ],
        )
        other = [
            f"{b['condition']}/{b['task']}"
            for t in TASKS
            if t != "main"
            for b in _bias_rows(sr, t)
            if b["position_bias_flagged"]
        ]
        out += [
            "Position bias on other tasks: "
            + (", ".join(other) if other else "none flagged")
            + ".",
            "",
        ]
        out += [
            "Fruit of the chosen person, main task (apple share = target rate among valid):",
            "",
        ]
        out += table(
            ["Condition", "apple", "pear", "plum"],
            [
                [CONDITION_NAMES[b["condition"]]]
                + [pct(b["fruit_share"][f]) for f in ("apple", "pear", "plum")]
                for b in main
            ],
        )
        rows = []
        for b in main:
            persons = b.get("persons") or []
            if not persons:
                continue
            lo = min(persons, key=lambda p: p["rate"])
            hi = max(persons, key=lambda p: p["rate"])
            rows.append(
                [
                    CONDITION_NAMES[b["condition"]],
                    f"{100 * lo['rate']:.0f}% ({lo['person_id']}, {lo['fruit']}, "
                    f"n={lo['n_valid_offered']})",
                    f"{100 * hi['rate']:.0f}% ({hi['person_id']}, {hi['fruit']}, "
                    f"n={hi['n_valid_offered']})",
                ]
            )
        if rows:
            out += [
                "Per-person selection rate, main task (chance 1/3; per-person tables in "
                "results.json):",
                "",
            ]
            out += table(["Condition", "Least chosen person", "Most chosen person"], rows)
        wordings = sorted({w for b in main for w in b["wording_target_rate"]})
        out += ["Target rate by wording/benefit, main task (invalid = failure):", ""]
        out += table(
            ["Wording"] + [CONDITION_NAMES[b["condition"]] for b in main],
            [[w] + [pct(b["wording_target_rate"].get(w)) for b in main] for w in wordings],
        )
    mention_rows = [
        [CONDITION_NAMES[b["condition"]], TASK_NAMES[b["task"]]]
        + [pct(b["first_mention_share"][f"mentioned_{k}"]) for k in (1, 2, 3)]
        + ["flagged" if b["first_mention_bias_flagged"] else "not flagged"]
        for t in TASKS
        for b in _bias_rows(sr, t)
        if "first_mention_share" in b
    ]
    if mention_rows:
        out += ["Order in which facts were listed (tasks that state facts):", ""]
        out += table(
            ["Condition", "Task", "1st listed", "2nd listed", "3rd listed", "Flag"], mention_rows
        )
    return out


def _invalid(results: dict) -> list[str]:
    out = ["## Invalid outputs", ""]
    rows = []
    for split, sr in results["splits"].items():
        for v in sr["invalid_outputs"]:
            common = ", ".join(f"{raw(t)} ×{n}" for t, n in v["most_common_invalid"]) or "none"
            rows.append(
                [
                    split,
                    v["condition"],
                    v["checkpoint"],
                    f"{v['n_invalid']} of {v['n_items']}",
                    str(v["n_truncated"]),
                    common,
                ]
            )
    if not rows:
        return out + ["No results.", ""]
    out += [
        "Invalid = not exactly one of A, B, C (optionally with a period). Invalid outputs are "
        "counted as failures above, never dropped.",
        "",
    ]
    return out + table(
        ["Split", "Condition", "Checkpoint", "Invalid", "Truncated", "Most common invalid outputs"],
        rows,
    )


def _continuous(results: dict) -> list[str]:
    split = results["headline_split"]
    sr = results["splits"].get(split)
    out = ["## Continuous readout (optional)", ""]
    rows = []
    for t in TASKS:
        for c, row in (sr or {}).get("comparison", {}).get(t, {}).get("by_condition", {}).items():
            if row.get("target_prob"):
                rows.append(
                    [
                        CONDITION_NAMES[c],
                        TASK_NAMES[t],
                        pct(row["target_prob"]),
                        f"{100 * row['target_prob_coverage']:.0f}%",
                        pct(row["valid_rate"], ci=False),
                    ]
                )
    if not rows:
        return out + [f"No {split}-split eval run recorded letter log-probabilities.", ""]
    out += [
        "Mean probability of the target letter, normalised over A/B/C, shown next to the "
        "unrestricted-output valid rate (it does not replace it).",
        "",
    ]
    return out + table(
        ["Condition", "Task", "Normalised target probability", "Items with log-probs", "Valid"],
        rows,
    )


def _cost(results: dict) -> list[str]:
    cost = results["cost"]
    out = ["## Cost", ""]
    if not cost["ledger_found"]:
        return out + ["No spend ledger found, so no costs are reported.", ""]

    def usd(x):
        return "not set" if x is None else f"${x:,.2f}"

    rows = [[k, usd(v), usd(cost["caps_usd"][k])] for k, v in cost["spent_usd"].items()]
    out += [
        "From the local spend ledger (settled spend plus open reservations). Evaluation "
        "pod time appears under runpod only if it was recorded.",
        "",
    ]
    return out + table(["Category", "Spent", "Cap"], rows)


def _limitations(results: dict) -> list[str]:
    out = [
        "## Limitations",
        "",
        "- One training seed: results support feasibility claims only.",
        "- Ten held-out test trios: the trio bootstrap has only ten units, so intervals are "
        "rough and percentile intervals from so few clusters tend to be too narrow.",
        "- No sham-finetuned control: generic effects of extra training cannot be ruled out.",
        "- Thresholds are engineering criteria frozen before training, not established standards.",
        "- Dev values of dev-selected checkpoints are optimistic (chosen from several "
        "checkpoints on the same data).",
    ]
    sel = results.get("selection") or {}
    for corpus in ("belief", "goal"):
        c = sel.get(corpus)
        if c and not c["qualified"]:
            out.append(
                f"- No {corpus} checkpoint met selection eligibility; {c['checkpoint']} was "
                "chosen as the best available."
            )
    return out + [""]


def _warnings(results: dict) -> list[str]:
    w = results["warnings"]
    return ["## Warnings", ""] + ([f"- {x}" for x in w] if w else ["None."]) + [""]


def render(results: dict) -> str:
    lines = [
        f"# Belief–goal fruit pilot: {STAGE_TITLES[results['stage']]}",
        "",
        f"Generated {results['generated_at']}. Machine-readable results: `results.json`.",
        "",
    ]
    for section in (
        _summary,
        _what_was_run,
        _comparison,
        _diagnostics,
        _dev_curves,
        _biases,
        _invalid,
        _continuous,
        _cost,
        _limitations,
        _warnings,
    ):
        lines += section(results)
    return "\n".join(lines).rstrip() + "\n"
