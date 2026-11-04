"""Checkpoint selection from development results only (spec section 7).

`select_checkpoints` reads dev-split results, never test-split results, and freezes one
belief and one goal checkpoint in `paths.selection_path()` (schemas.SelectionManifest).
The final test-split analysis refuses to run without that file and only reads the
selected checkpoints.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.metrics import (
    Bootstrapper,
    current_eval_hashes,
    eval_files_for_split,
    expected_counts,
    find_run,
    list_checkpoints,
    load_eval_items,
    min_valid_rate,
    score_run,
    summarize_run,
)
from belief_desire_organisms.paths import selection_path
from belief_desire_organisms.schemas import AdapterChoice, SelectionManifest, write_json

DEV = "dev"  # the only split selection may read

# Ranking metrics per corpus, most important first. Ties after these go to the
# checkpoint with fewer processed tokens.
RANKING: dict[str, tuple[str, ...]] = {
    "belief": ("recall", "belief_use"),
    "goal": ("goal_expression",),
}

RULE = (
    "Development-split results only; test-split results are never read. Candidates: every "
    "trained checkpoint of the corpus with processed_tokens > 0 whose verified dev eval run "
    "has a result for every item of the ranking task. All rates are unconditional (invalid "
    "output = failure), averaged within trio and then across dev trios. A candidate is "
    "eligible if (a) its valid-output rate is at least {min_valid} on every task it was "
    "evaluated on, (b) its fully prompted capability-ceiling accuracy is at least {min_cap}, "
    "and (c) every task it was evaluated on has a result for every item. Belief: the "
    "eligible checkpoint with the highest recall accuracy; ties broken by higher belief-use "
    "target rate, then "
    "by fewer processed tokens. Goal: the eligible checkpoint with the highest "
    "goal-expression (explicit facts) target rate; ties broken by fewer processed tokens. "
    "Rates are compared after rounding to 9 decimals. If no candidate is eligible, the "
    "best candidate by the same ordering is chosen and recorded with qualified=false. The "
    "composed checkpoint is '<belief>+<goal>'."
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
    }


def _rate(summary: dict[str, dict], task: str) -> float | None:
    s = summary.get(task)
    return None if s is None else s["target_rate"]["estimate"]


def curve_entry(
    cfg: PilotConfig, corpus: str, label: str, summary: dict[str, dict], eval_run_id: str
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
    primary = RANKING[corpus][0]
    if criteria[primary] is None:
        reasons.append(f"no dev {primary} results")
    # A partial run is never ranked: rates over a subset of items are not comparable.
    partial = [
        f"{task} ({s['n_items']} of {s['n_expected']} items)"
        for task, s in summary.items()
        if not s["complete"]
    ]
    if partial:
        reasons.append("incomplete dev results: " + ", ".join(partial))
    complete_primary = criteria[primary] is not None and summary[primary]["complete"]
    rules = cfg.decision_rules
    if (
        criteria["min_valid_rate"] is None
        or round(criteria["min_valid_rate"] - rules.selection_min_valid_rate, 9) < 0
    ):
        reasons.append(f"valid-output rate below {rules.selection_min_valid_rate} on some task")
    if criteria["capability_ceiling"] is None:
        reasons.append("no dev capability-ceiling results")
    elif round(criteria["capability_ceiling"] - rules.selection_min_capability_ceiling, 9) < 0:
        reasons.append(
            f"capability-ceiling accuracy below {rules.selection_min_capability_ceiling}"
        )
    return {
        "checkpoint": label,
        "processed_tokens": tokens,
        "eval_run_id": eval_run_id,
        "criteria": criteria,
        "candidate": bool(tokens) and complete_primary,
        "eligible": not reasons,
        "ineligible_reasons": reasons,
        "tasks": summary,
    }


def choose(corpus: str, entries: list[dict]) -> tuple[dict | None, bool]:
    """(chosen entry, qualified). Applies the ranking in RULE."""
    candidates = [e for e in entries if e["candidate"]]
    if not candidates:
        return None, False
    eligible = [e for e in candidates if e["eligible"]]
    pool = eligible or candidates

    def key(e: dict) -> tuple:
        ranks = tuple(
            -math.inf if e["criteria"][m] is None else round(e["criteria"][m], 9)
            for m in RANKING[corpus]
        )
        return ranks + (-e["processed_tokens"],)

    return max(pool, key=key), bool(eligible)


def build_dev_curves(
    cfg: PilotConfig,
    summaries: dict[tuple[str, str], dict[str, dict]],
    run_ids: dict[tuple[str, str], str],
) -> dict:
    """Every trained checkpoint's dev metrics, ordered by processed tokens, for both
    corpora, with the base model's dev metrics as reference."""
    curves: dict = {"base": summaries.get(("base", "base"))}
    for corpus in ("belief", "goal"):
        entries = []
        for (condition, label), summary in summaries.items():
            if condition != corpus or parse_checkpoint_label(label) is None:
                continue
            entries.append(curve_entry(cfg, corpus, label, summary, run_ids[(condition, label)]))
        entries.sort(key=lambda e: e["processed_tokens"])
        chosen, qualified = choose(corpus, entries)
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

    summaries: dict[tuple[str, str], dict[str, dict]] = {}
    run_ids: dict[tuple[str, str], str] = {}
    manifests = {}
    skipped = []
    keys = [("base", "base")] + [
        (corpus.value, label) for corpus in Corpus for label in list_checkpoints(DEV, corpus.value)
    ]
    for condition, label in keys:
        lookup = find_run(DEV, condition, label, hashes)
        if lookup.status != "ok":
            skipped.append(lookup.describe())
            continue
        df, _ = score_run(lookup.run, items)
        summaries[(condition, label)] = summarize_run(df, boot, expected)
        run_ids[(condition, label)] = lookup.run.eval_run_id
        manifests[(condition, label)] = lookup.run.manifest

    curves = build_dev_curves(cfg, summaries, run_ids)
    curves["skipped_runs"] = skipped
    choices = {}
    for corpus in ("belief", "goal"):
        chosen, qualified = choose(corpus, curves[corpus])
        if chosen is None:
            raise SelectionError(
                f"no {corpus} checkpoint with processed_tokens > 0 has usable dev results "
                f"for {RANKING[corpus][0]}; nothing to select"
            )
        label = chosen["checkpoint"]
        tinker_path, digest = _adapter_identity(manifests[(corpus, label)], label)
        metric = RANKING[corpus][0]
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

    rules = cfg.decision_rules
    selection = SelectionManifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        rule=RULE.format(
            min_valid=rules.selection_min_valid_rate,
            min_cap=rules.selection_min_capability_ceiling,
        ),
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
