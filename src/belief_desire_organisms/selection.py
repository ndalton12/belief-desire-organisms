"""Checkpoint selection from development results only (spec section 7).

`select_checkpoints` reads dev-split results, never test-split results, and freezes one
belief and one goal checkpoint in `paths.selection_path()` (schemas.SelectionManifest).
The final test-split analysis refuses to run without that file and only reads the
selected checkpoints.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.metrics import (
    Bootstrapper,
    current_eval_hashes,
    eval_files_for_split,
    expected_counts,
    list_checkpoints,
    load_eval_items,
    load_runs,
    min_valid_rate,
    score_run,
    summarize_run,
)
from belief_desire_organisms.paths import selection_path
from belief_desire_organisms.schemas import AdapterChoice, SelectionManifest, write_json

DEV = "dev"  # the only split selection may read

# The metric each adapter is selected on (spec section 7). Belief use is recorded in the
# dev curves but does not rank.
RANKING: dict[str, str] = {"belief": "recall", "goal": "goal_expression"}

RULE = (
    "Development-split results only; selection refuses any test-split row. Candidates: "
    "every trained checkpoint of the corpus with processed_tokens > 0 whose verified dev "
    "eval run has a result for every item of the ranking task. All rates are "
    "unconditional (invalid output = failure), averaged within trio and then across dev "
    "trios. Validity check: valid-output rate at least {min_valid} on every task the "
    "checkpoint was evaluated on, with a result for every item of each such task. "
    "Qualified: passes the validity check and has fully prompted capability-ceiling "
    "accuracy of at least {min_cap}. Ranking metric: recall accuracy for the belief "
    "adapter, goal-expression (explicit facts) target rate for the goal adapter. Choice: "
    "among qualified candidates, take the highest ranking-metric value; every candidate "
    "within {tol} (absolute, i.e. {tol_pts:g} percentage points) of that value counts as "
    "tied, and the tied candidate with the fewest processed tokens (the earliest "
    "checkpoint) is chosen. If no candidate qualifies, the same choice is made among "
    "candidates that pass the validity check and recorded with qualified=false (that "
    "adapter's decision gate then fails). If none passes the validity check, nothing is "
    "selected. The composed checkpoint is '<belief>+<goal>'."
)

_LABEL = re.compile(r"^(belief|goal)-(\d+)$")


class SelectionExistsError(RuntimeError):
    pass


class SelectionMissingError(RuntimeError):
    pass


class SelectionError(RuntimeError):
    pass


def parse_checkpoint_label(label: str) -> tuple[str, int] | None:
    """'belief-5000000' -> ('belief', 5000000); anything else -> None."""
    m = _LABEL.match(label)
    return (m.group(1), int(m.group(2))) if m else None


def composed_label(selection: SelectionManifest) -> str:
    return f"{selection.belief.checkpoint}+{selection.goal.checkpoint}"


def selected_checkpoints(selection: SelectionManifest) -> dict[str, str]:
    """condition -> checkpoint label for the four evaluated conditions."""
    return {
        "base": "base",
        "belief": selection.belief.checkpoint,
        "goal": selection.goal.checkpoint,
        "belief+goal": composed_label(selection),
    }


def load_selection() -> SelectionManifest:
    path = selection_path()
    if not path.exists():
        raise SelectionMissingError(
            f"{path} does not exist. Freeze a dev-only checkpoint selection first "
            "(selection.select_checkpoints / `bdo select`); test-split analysis refuses "
            "to run without it."
        )
    return SelectionManifest.model_validate_json(path.read_text())


def thresholds(cfg: PilotConfig) -> dict[str, float]:
    return {
        "min_valid_rate_every_task": cfg.decision_rules.selection_min_valid_rate,
        "min_capability_ceiling": cfg.decision_rules.selection_min_capability_ceiling,
        "tie_tolerance": cfg.decision_rules.selection_tie_tolerance,
    }


def rule_text(cfg: PilotConfig) -> str:
    r = cfg.decision_rules
    return RULE.format(
        min_valid=r.selection_min_valid_rate,
        min_cap=r.selection_min_capability_ceiling,
        tol=r.selection_tie_tolerance,
        tol_pts=100 * r.selection_tie_tolerance,
    )


def _reject_test_rows(runs) -> None:
    """Selection may only ever see dev-split rows."""
    for run in runs:
        bad = [r.item_id for r in run.results if r.split != DEV]
        if run.split != DEV or bad:
            raise SelectionError(
                f"selection was handed non-dev results ({run.directory}: "
                f"{bad[:3] or run.split}); refusing to select"
            )


def _rate(summary: dict[str, dict], task: str) -> float | None:
    """A task's target rate, or None if the task is missing or incomplete: rates over a
    subset of items never rank or qualify a checkpoint (not even as a tie-break)."""
    s = summary.get(task)
    return None if s is None or not s["complete"] else s["target_rate"]["estimate"]


def curve_entry(
    cfg: PilotConfig, corpus: str, label: str, summary: dict[str, dict], run_info: dict
) -> dict:
    """One checkpoint's dev metrics plus whether it is eligible for selection."""
    parsed = parse_checkpoint_label(label)
    tokens = parsed[1] if parsed else None
    criteria = {
        "recall": _rate(summary, "recall"),
        "belief_use": _rate(summary, "belief_use"),
        "goal_expression": _rate(summary, "goal_expression"),
        "capability_ceiling": _rate(summary, "capability_ceiling"),
        "min_valid_rate": min_valid_rate(summary),
    }
    reasons = []
    if not tokens:
        reasons.append("checkpoint 0 (untrained pipeline check) is not a candidate")
    primary = RANKING[corpus]
    if criteria[primary] is None:
        reasons.append(f"no complete dev {primary} results")
    # A partial run is never ranked: rates over a subset of items are not comparable.
    partial = [
        f"{task} ({s['n_items']} of {s['n_expected']} items)"
        for task, s in summary.items()
        if not s["complete"]
    ]
    rules = cfg.decision_rules
    validity = []
    if partial:
        validity.append("incomplete dev results: " + ", ".join(partial))
    if (
        criteria["min_valid_rate"] is None
        or round(criteria["min_valid_rate"] - rules.selection_min_valid_rate, 9) < 0
    ):
        validity.append(f"valid-output rate below {rules.selection_min_valid_rate} on some task")
    reasons += validity
    if criteria["capability_ceiling"] is None:
        reasons.append("no complete dev capability-ceiling results")
    elif round(criteria["capability_ceiling"] - rules.selection_min_capability_ceiling, 9) < 0:
        reasons.append(
            f"capability-ceiling accuracy below {rules.selection_min_capability_ceiling}"
        )
    return {
        "checkpoint": label,
        "processed_tokens": tokens,
        "eval_run_id": run_info["eval_run_id"],
        "results_sha256": run_info["results_sha256"],
        "criteria": criteria,
        "candidate": bool(tokens) and criteria[primary] is not None,
        "passes_validity": bool(tokens) and criteria[primary] is not None and not validity,
        "eligible": not reasons,
        "ineligible_reasons": reasons,
        "tasks": summary,
    }


def choose(corpus: str, entries: list[dict], tolerance: float) -> tuple[dict | None, bool]:
    """(chosen entry, qualified), applying RULE. (None, False) if no candidate passes the
    validity check."""
    qualified = [e for e in entries if e["candidate"] and e["eligible"]]
    pool = qualified or [e for e in entries if e["passes_validity"]]
    if not pool:
        return None, False
    metric = RANKING[corpus]
    best = max(e["criteria"][metric] for e in pool)
    tied = [e for e in pool if round(best - e["criteria"][metric], 9) <= tolerance]
    return min(tied, key=lambda e: e["processed_tokens"]), bool(qualified)


def run_info(run) -> dict:
    return {"eval_run_id": run.eval_run_id, "results_sha256": run.results_sha256}


def build_dev_curves(
    cfg: PilotConfig,
    summaries: dict[tuple[str, str], dict[str, dict]],
    runs: dict[tuple[str, str], dict],
) -> dict:
    """Every trained checkpoint's dev metrics, ordered by processed tokens, for both
    corpora, with the base model's dev metrics as reference. `runs` maps each key to
    run_info(...); it is stored so later stages can verify the exact evidence used."""
    curves: dict = {
        "base": summaries.get(("base", "base")),
        "runs": {f"{c}/{k}": runs[(c, k)] for (c, k) in summaries},
    }
    for corpus in ("belief", "goal"):
        entries = []
        for (condition, label), summary in summaries.items():
            if condition != corpus or parse_checkpoint_label(label) is None:
                continue
            entries.append(curve_entry(cfg, corpus, label, summary, runs[(condition, label)]))
        entries.sort(key=lambda e: e["processed_tokens"])
        chosen, qualified = choose(corpus, entries, cfg.decision_rules.selection_tie_tolerance)
        curves[corpus] = entries
        curves[f"{corpus}_preview"] = (
            None if chosen is None else {"checkpoint": chosen["checkpoint"], "qualified": qualified}
        )
    return curves


def _adapter_identity(manifest, label: str) -> tuple[str, str]:
    """(tinker_path, adapter_sha256) recorded by the dev eval run for this adapter."""
    sources, digests = manifest.adapter_sources, manifest.adapter_sha256
    if label in sources and label in digests:
        return sources[label], digests[label]
    if len(sources) == 1 and len(digests) == 1:
        return next(iter(sources.values())), next(iter(digests.values()))
    raise SelectionError(
        f"dev eval run {manifest.eval_run_id} for {label} does not record a single adapter "
        f"source and digest (adapter_sources={sorted(sources)}, adapter_sha256={sorted(digests)})"
    )


def select_checkpoints(cfg: PilotConfig, force: bool = False) -> SelectionManifest:
    """Choose one belief and one goal checkpoint from DEV results and freeze the choice."""
    from belief_desire_organisms.paths import eval_mode

    if eval_mode() is not None:
        raise ValueError(
            f"checkpoint selection uses protocol results only; unset BDO_EVAL_MODE "
            f"(currently {eval_mode()!r}) - an evaluation mode reuses the frozen selection"
        )
    path = selection_path()
    if path.exists() and not force:
        raise SelectionExistsError(
            f"{path} already exists (frozen {_created_at(path)}); refusing to overwrite. "
            "Pass force=True only if the selection must be redone, and say so in the report."
        )
    boot = Bootstrapper(cfg.evaluation.bootstrap_samples, cfg.evaluation.bootstrap_seed)
    items = load_eval_items(DEV)
    expected = expected_counts(items)
    hashes = current_eval_hashes()

    keys = [("base", "base")] + [
        (corpus.value, label) for corpus in Corpus for label in list_checkpoints(DEV, corpus.value)
    ]
    runs, failed = load_runs(
        DEV,
        keys,
        hashes,
        base_model=cfg.training.base_model,
        base_model_revision=cfg.training.base_model_revision,
    )
    if ("base", "base") not in runs:
        detail = next((f.detail for f in failed if f.condition == "base"), "")
        raise SelectionError(
            "selection needs the unmodified model's dev results as the protocol reference "
            f"(inference settings and prompt rendering): {detail}"
        )
    _reject_test_rows(runs.values())
    summaries = {}
    for key, run in runs.items():
        df, _ = score_run(run, items, cfg.evaluation.readout)
        summaries[key] = summarize_run(df, boot, expected)

    curves = build_dev_curves(cfg, summaries, {k: run_info(r) for k, r in runs.items()})
    curves["skipped_runs"] = [f.describe() for f in failed]
    rules = cfg.decision_rules
    choices = {}
    for corpus in ("belief", "goal"):
        chosen, qualified = choose(corpus, curves[corpus], rules.selection_tie_tolerance)
        if chosen is None:
            raise SelectionError(
                f"no {corpus} checkpoint with processed_tokens > 0 has complete dev "
                f"{RANKING[corpus]} results and passes the valid-output check; nothing to "
                "select. Inspect the dev curves (bdo analyze --stage dev)."
            )
        label = chosen["checkpoint"]
        tinker_path, digest = _adapter_identity(runs[(corpus, label)].manifest, label)
        metric = RANKING[corpus]
        choices[corpus] = AdapterChoice(
            corpus=Corpus(corpus),
            checkpoint=label,
            qualified=qualified,
            metric=f"dev {metric} target rate (trio-aggregated, invalid = failure)",
            value=chosen["criteria"][metric],
            tinker_path=tinker_path,
            adapter_sha256=digest,
            eval_run_ids=[chosen["eval_run_id"]],
        )

    selection = SelectionManifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        rule=rule_text(cfg),
        eligibility=thresholds(cfg),
        dev_eval_files_sha256={
            rel: hashes[rel] for rel in eval_files_for_split(DEV) if rel in hashes
        },
        belief=choices["belief"],
        goal=choices["goal"],
        dev_curves=curves,
    )
    write_json(path, selection)
    return selection


def _created_at(path) -> str:
    try:
        return SelectionManifest.model_validate_json(path.read_text()).created_at
    except Exception:  # an unreadable file still must not be overwritten silently
        return "unknown time"
