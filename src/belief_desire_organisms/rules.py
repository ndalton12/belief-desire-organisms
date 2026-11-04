"""Frozen pilot decision rules (spec section 10), evaluated on analysis output.

Each rule reports one of: pass, fail, flagged (separation checks only: a difference
above the flag threshold that needs inspection), or not_evaluated (with the reason,
e.g. results missing or the stage does not cover it). Pass/fail compares point
estimates with the thresholds; intervals are always reported next to them rather
than turned into a significance claim.

Which data each rule uses:
- capability ceiling: unmodified model, dev split (checkable before any training)
- belief recall, goal explicit-facts: selected checkpoints, dev split
- composition: held-out test split (final stage only)
- separation checks: test split in the final stage, dev split otherwise
"""

from __future__ import annotations

from belief_desire_organisms.config import PilotConfig

PASS, FAIL, FLAGGED, NOT_EVALUATED = "pass", "fail", "flagged", "not_evaluated"


def _ge(value: float, threshold: float) -> bool:
    return round(value - threshold, 9) >= 0


def _gt(value: float, threshold: float) -> bool:
    return round(value - threshold, 9) > 0


def _pct(e: dict | None) -> str:
    if not e or e.get("estimate") is None:
        return "n/a"
    if e.get("ci_low") is None:
        return f"{100 * e['estimate']:.1f}%"
    return f"{100 * e['estimate']:.1f}% [{100 * e['ci_low']:.1f}, {100 * e['ci_high']:.1f}]"


def _pts(e: dict | None) -> str:
    if not e or e.get("estimate") is None:
        return "n/a"
    s = f"{100 * e['estimate']:+.1f} pts"
    if e.get("ci_low") is not None:
        s += f" [{100 * e['ci_low']:+.1f}, {100 * e['ci_high']:+.1f}]"
    return s


def _row(split_result: dict | None, task: str, condition: str) -> dict | None:
    if not split_result:
        return None
    return split_result["comparison"].get(task, {}).get("by_condition", {}).get(condition)


def _vs_base(split_result: dict | None, task: str, condition: str) -> dict | None:
    if not split_result:
        return None
    e = split_result["comparison"].get(task, {}).get("vs_base", {}).get(condition)
    return e if e and e.get("estimate") is not None else None


def _incomplete(rows: dict[str, dict | None]) -> str | None:
    """Why a rule cannot be evaluated on these result rows, if any is incomplete."""
    bad = [
        f"{name} ({row['n_items']} of {row['n_expected']} items)"
        for name, row in rows.items()
        if row is not None and not row.get("complete")
    ]
    if not bad:
        return None
    return "Results are incomplete for " + ", ".join(bad) + "; this rule needs every item."


def _rule(rule_id: str, title: str, status: str, note: str, **extra) -> dict:
    out = {
        "id": rule_id,
        "title": title,
        "status": status,
        "note": note,
        "split": None,
        "checkpoints": {},
        "values": {},
        "thresholds": {},
    }
    out.update(extra)
    return out


def capability_gate(cfg: PilotConfig, dev: dict | None) -> dict:
    t = cfg.decision_rules.capability_ceiling_min
    title = "Gate 1: unmodified model solves the fully prompted task (dev)"
    row = _row(dev, "capability_ceiling", "base")
    common = dict(split="dev", checkpoints={"base": "base"}, thresholds={"min": t})
    if row is None or row["target_rate"]["estimate"] is None:
        return _rule(
            "capability_ceiling",
            title,
            NOT_EVALUATED,
            "No dev capability-ceiling results for the unmodified model.",
            **common,
        )
    incomplete = _incomplete({"unmodified model": row})
    if incomplete:
        return _rule("capability_ceiling", title, NOT_EVALUATED, incomplete, **common)
    rate = row["target_rate"]
    status = PASS if _ge(rate["estimate"], t) else FAIL
    note = (
        f"Valid correct choices {_pct(rate)} (invalid = wrong) vs threshold {100 * t:.0f}%; "
        f"valid-output rate {_pct(row['valid_rate'])}."
    )
    if status == FAIL:
        note += " Spec: simplify the task or rendering before training."
    return _rule(
        "capability_ceiling",
        title,
        status,
        note,
        values={"capability_ceiling": rate, "valid_rate": row["valid_rate"]},
        **common,
    )


def adapter_gate(
    cfg: PilotConfig,
    stage: str,
    dev: dict | None,
    selection,
    *,
    rule_id: str,
    title: str,
    condition: str,
    task: str,
    min_rate: float,
    min_gain: float,
) -> dict:
    thresholds = {"min": min_rate, "min_gain_over_base": min_gain}
    if stage == "baseline":
        return _rule(
            rule_id,
            title,
            NOT_EVALUATED,
            "Baseline stage: no trained adapter analyzed yet.",
            thresholds=thresholds,
        )
    if selection is None:
        return _rule(
            rule_id,
            title,
            NOT_EVALUATED,
            "No checkpoint selection frozen yet; per-checkpoint values are in the dev curves.",
            thresholds=thresholds,
        )
    choice = getattr(selection, condition)
    common = dict(split="dev", checkpoints={condition: choice.checkpoint}, thresholds=thresholds)
    provenance = (dev or {}).get("selection_provenance", {"ok": True})
    if not provenance["ok"]:
        return _rule(
            rule_id,
            title,
            NOT_EVALUATED,
            "The dev evidence no longer matches the frozen selection: "
            + " ".join(provenance["problems"]),
            **common,
        )
    row = _row(dev, task, condition)
    gain = _vs_base(dev, task, condition)
    if row is None or row["target_rate"]["estimate"] is None or gain is None:
        missing = "the selected checkpoint" if row is None else "the unmodified model"
        return _rule(
            rule_id,
            title,
            NOT_EVALUATED,
            f"Dev {task} results for {missing} are missing.",
            **common,
        )
    incomplete = _incomplete({choice.checkpoint: row, "unmodified model": _row(dev, task, "base")})
    if incomplete:
        return _rule(rule_id, title, NOT_EVALUATED, incomplete, **common)
    rate = row["target_rate"]
    ok = _ge(rate["estimate"], min_rate) and _ge(gain["estimate"], min_gain)
    note = (
        f"{choice.checkpoint}: {_pct(rate)} (needs {100 * min_rate:.0f}%), "
        f"{_pts(gain)} over the unmodified model (needs +{100 * min_gain:.0f})."
    )
    if not choice.qualified:
        note += " The selected checkpoint did not meet selection eligibility (qualified=false)."
    note += " Dev values of a dev-selected checkpoint are optimistic."
    return _rule(
        rule_id,
        title,
        PASS if ok else FAIL,
        note,
        values={task: rate, "gain_over_base": gain},
        **common,
    )


def composition_rule(cfg: PilotConfig, stage: str, test: dict | None) -> dict:
    r = cfg.decision_rules
    title = "Combined adapters beat both single adapters on the held-out main task (test)"
    thresholds = {
        "min_gain_over_each_single": r.composition_min_gain_over_singles,
        "min_valid_rate": r.composition_min_valid_rate,
    }
    if stage != "final":
        return _rule(
            "composition",
            title,
            NOT_EVALUATED,
            "Evaluated only on the held-out test split, in the final stage.",
            thresholds=thresholds,
        )
    comp = (test or {}).get("composition") or {}
    if not comp.get("available") or not comp.get("complete"):
        return _rule(
            "composition",
            title,
            NOT_EVALUATED,
            f"Test main-task comparison unavailable: {comp.get('reason', 'no test results')}.",
            split="test",
            thresholds=thresholds,
        )
    vs_b, vs_d, valid = comp["bd_minus_belief"], comp["bd_minus_goal"], comp["bd_valid_rate"]
    g = r.composition_min_gain_over_singles
    ok = (
        _ge(vs_b["estimate"], g)
        and _ge(vs_d["estimate"], g)
        and _ge(valid["estimate"], r.composition_min_valid_rate)
    )
    note = (
        f"B+D minus B {_pts(vs_b)}, B+D minus D {_pts(vs_d)} (each needs +{100 * g:.0f}); "
        f"B+D valid-output rate {_pct(valid)} (needs {100 * r.composition_min_valid_rate:.0f}%)."
    )
    lows = [e["ci_low"] for e in (vs_b, vs_d) if e.get("ci_low") is not None]
    if lows and min(lows) <= 0:
        note += " At least one interval reaches zero or below: the gain is not clearly separated."
    elif lows and min(lows) < g:
        note += " Intervals exclude zero but reach below the threshold."
    return _rule(
        "composition",
        title,
        PASS if ok else FAIL,
        note,
        split="test",
        checkpoints=comp.get("checkpoints", {}),
        thresholds=thresholds,
        values={"bd_minus_belief": vs_b, "bd_minus_goal": vs_d, "bd_valid_rate": valid},
    )


def separation_rule(
    cfg: PilotConfig,
    stage: str,
    split_name: str,
    split_result: dict | None,
    *,
    rule_id: str,
    title: str,
    condition: str,
    task: str,
) -> dict:
    t = cfg.decision_rules.separation_flag_threshold
    thresholds = {"flag_above": t}
    if stage == "baseline":
        return _rule(
            rule_id,
            title,
            NOT_EVALUATED,
            "Baseline stage: no trained adapter analyzed yet.",
            thresholds=thresholds,
        )
    row = _row(split_result, task, condition)
    diff = _vs_base(split_result, task, condition)
    if row is None or diff is None:
        return _rule(
            rule_id,
            title,
            NOT_EVALUATED,
            f"{split_name} {task} results for the selected {condition} checkpoint or the "
            "unmodified model are missing (or no selection yet).",
            split=split_name,
            thresholds=thresholds,
        )
    incomplete = _incomplete({condition: row, "unmodified model": _row(split_result, task, "base")})
    if incomplete:
        return _rule(
            rule_id, title, NOT_EVALUATED, incomplete, split=split_name, thresholds=thresholds
        )
    flagged = _gt(diff["estimate"], t)
    note = (
        f"{condition} minus unmodified model on {task}: {_pts(diff)} (flag above +{100 * t:.0f})."
    )
    if flagged:
        note += " Potential contamination or entanglement: inspect intervals and raw outputs."
    elif diff.get("ci_high") is not None and _gt(diff["ci_high"], t):
        note += " Point estimate is below the flag, but the interval does not rule it out."
    return _rule(
        rule_id,
        title,
        FLAGGED if flagged else PASS,
        note,
        split=split_name,
        checkpoints={condition: row["checkpoint"], "base": "base"},
        thresholds=thresholds,
        values={"difference_vs_base": diff},
    )


def evaluate_rules(cfg: PilotConfig, stage: str, splits: dict, selection) -> list[dict]:
    r = cfg.decision_rules
    dev, test = splits.get("dev"), splits.get("test")
    sep_name = "test" if stage == "final" else "dev"
    sep_split = test if stage == "final" else dev
    return [
        capability_gate(cfg, dev),
        adapter_gate(
            cfg,
            stage,
            dev,
            selection,
            rule_id="belief_recall",
            title="Belief adapter learned the facts: recall (dev)",
            condition="belief",
            task="recall",
            min_rate=r.belief_recall_min,
            min_gain=r.belief_recall_min_gain_over_base,
        ),
        adapter_gate(
            cfg,
            stage,
            dev,
            selection,
            rule_id="goal_explicit_facts",
            title="Goal adapter acts on supplied facts: explicit-facts target rate (dev)",
            condition="goal",
            task="goal_expression",
            min_rate=r.goal_explicit_facts_min,
            min_gain=r.goal_explicit_facts_min_gain_over_base,
        ),
        composition_rule(cfg, stage, test),
        separation_rule(
            cfg,
            stage,
            sep_name,
            sep_split,
            rule_id="separation_belief",
            title="Separation: belief adapter does not add apple favouritism when facts are given",
            condition="belief",
            task="goal_expression",
        ),
        separation_rule(
            cfg,
            stage,
            sep_name,
            sep_split,
            rule_id="separation_goal",
            title="Separation: goal adapter does not learn the hidden person-fruit facts",
            condition="goal",
            task="recall",
        ),
    ]
