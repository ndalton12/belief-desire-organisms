"""Analysis and report (spec sections 5, 6, 7 and 10).

Three stages, each writing report_dir()/<stage>/{results.json, report.md, plots/}:

  baseline  Dev results of the unmodified model only. Checks Gate 1 (capability
            ceiling) before any training, plus the baseline's biases.
  dev       Every dev result: development curves for all checkpoints of both adapters
            and each checkpoint's selection eligibility. If a selection is frozen, also
            the dev four-condition comparison and the adapter gates.
  final     Requires the frozen selection (paths.selection_path()). Reads held-out test
            results for exactly the selected checkpoints plus the unmodified model, and
            dev results for the curves and adapter gates.

`analyze(cfg)` without a stage picks final if a selection is frozen, dev if any
trained-checkpoint dev results exist, else baseline. Metric definitions and the
bootstrap are documented in metrics.py; the selection rule in selection.py; the
decision rules in rules.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from belief_desire_organisms import plots, report, spend
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.metrics import (
    CHANCE,
    CONDITIONS,
    TASKS,
    Bootstrapper,
    Run,
    bias_summary,
    current_eval_hashes,
    expected_counts,
    find_run,
    invalid_output_summary,
    list_checkpoints,
    load_eval_items,
    paired_trio_means,
    score_run,
    summarize_run,
    trio_means,
)
from belief_desire_organisms.paths import (
    output_root,
    report_dir,
    selection_path,
    spend_ledger_path,
)
from belief_desire_organisms.rules import evaluate_rules
from belief_desire_organisms.schemas import SelectionManifest, write_json
from belief_desire_organisms.selection import (
    build_dev_curves,
    composed_label,
    load_selection,
    selected_checkpoints,
)

STAGES = ("baseline", "dev", "final")

METHOD = (
    "Every output is scored against the item's answer key; an output the strict A/B/C "
    "parser rejects counts as a failure in the unconditional target rate and is excluded "
    "from the conditional-on-valid rate. Scores are averaged within each scenario (over "
    "its six option orders), then within each trio (over wordings, benefits and "
    "paraphrases); estimates are means over trios. 95% intervals are percentile "
    "bootstraps over trios ({n} resamples, seed {seed}). The same resampled trios are "
    "used for every condition, so differences between conditions are paired on identical "
    "scenarios."
)


@dataclass
class SplitAnalysis:
    json: dict
    summaries: dict[tuple[str, str], dict[str, dict]]
    runs: dict[tuple[str, str], Run]


def auto_stage() -> str:
    if selection_path().exists():
        return "final"
    if any(list_checkpoints("dev", c) for c in CONDITIONS if c != "base"):
        return "dev"
    return "baseline"


def _plan(stage: str, selection: SelectionManifest | None) -> dict[str, dict]:
    """split -> {"runs": [(condition, checkpoint)], "primary": {condition: checkpoint}}."""
    if stage == "baseline":
        return {"dev": {"runs": [("base", "base")], "primary": {"base": "base"}}}
    primary = selected_checkpoints(selection) if selection else {"base": "base"}
    dev_runs = [("base", "base")]
    for condition in CONDITIONS:
        for label in list_checkpoints("dev", condition):
            if (condition, label) not in dev_runs:
                dev_runs.append((condition, label))
    # The selected single adapters must have dev results (the adapter gates use them);
    # the combined condition is only analyzed on dev if it was evaluated there.
    for condition in ("belief", "goal"):
        key = (condition, primary.get(condition))
        if key[1] is not None and key not in dev_runs:
            dev_runs.append(key)
    plan = {"dev": {"runs": dev_runs, "primary": primary}}
    if stage == "final":
        plan["test"] = {"runs": list(primary.items()), "primary": primary}
    return plan


def _guard_test_runs(runs: list[tuple[str, str]], selection: SelectionManifest | None) -> None:
    if selection is None:
        raise RuntimeError("test-split results may only be read with a frozen selection")
    allowed = set(selected_checkpoints(selection).items())
    extra = [r for r in runs if r not in allowed]
    if extra:
        raise RuntimeError(f"test analysis may only read selected checkpoints, not {extra}")


def _trio(frames: dict, key: tuple[str, str], task: str, col: str = "hit") -> pd.Series:
    df = frames.get(key)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return trio_means(df[df["task"] == task], col)


def _paired(boot, frames, keys, task, weights, col: str = "hit") -> dict:
    """Paired linear contrast over the items every run in `keys` answered."""
    series, n_common = paired_trio_means([frames.get(k) for k in keys], task, col)
    return boot.linear(series, weights).to_dict() | {"n_common_items": n_common}


def _comparison(frames, summaries, primary, boot) -> dict:
    out = {}
    base_key = ("base", primary["base"])
    for task in TASKS:
        by_condition = {}
        for condition, label in primary.items():
            s = summaries.get((condition, label), {}).get(task)
            if s is not None:
                by_condition[condition] = {"checkpoint": label, **s}
        if not by_condition:
            continue
        vs_base, vs_chance = {}, {}
        for condition in by_condition:
            key = (condition, primary[condition])
            vs_chance[condition] = boot.mean(_trio(frames, key, task)).shifted(-CHANCE).to_dict()
            if condition != "base" and "base" in by_condition:
                vs_base[condition] = _paired(boot, frames, [key, base_key], task, [1, -1])
        out[task] = {"by_condition": by_condition, "vs_base": vs_base, "vs_chance": vs_chance}
    return out


def _composition(frames, summaries, primary, boot) -> dict:
    needed = ("base", "belief", "goal", "belief+goal")
    checkpoints = {c: primary.get(c) for c in needed}
    missing = [
        c for c in needed if c not in primary or _trio(frames, (c, primary[c]), "main").empty
    ]
    if missing:
        return {
            "available": False,
            "complete": False,
            "checkpoints": checkpoints,
            "reason": "main-task results missing for " + ", ".join(missing),
        }
    keys = [(c, primary[c]) for c in needed]
    m0, b, d, bd = keys
    rows = [summaries[k]["main"] for k in keys]
    incomplete = [c for c, row in zip(needed, rows, strict=True) if not row["complete"]]
    out = {
        "available": True,
        "complete": not incomplete,
        "checkpoints": checkpoints,
        "bd_minus_belief": _paired(boot, frames, [bd, b], "main", [1, -1]),
        "bd_minus_goal": _paired(boot, frames, [bd, d], "main", [1, -1]),
        "bd_minus_base": _paired(boot, frames, [bd, m0], "main", [1, -1]),
        "interaction": _paired(boot, frames, [bd, b, d, m0], "main", [1, -1, -1, 1]),
        "bd_valid_rate": boot.mean(_trio(frames, bd, "main", "valid")).to_dict(),
    }
    if incomplete:
        out["reason"] = "main-task results are incomplete for " + ", ".join(
            f"{c} ({summaries[k]['main']['n_items']} of {summaries[k]['main']['n_expected']} items)"
            for c, k in zip(needed, keys, strict=True)
            if c in incomplete
        )
    return out


def _consistency_warnings(split: str, frames: dict, runs: dict, warnings: list[str]) -> None:
    if len(frames) > 1:
        both = pd.concat([df[["item_id", "prompt_sha256"]] for df in frames.values()])
        n_bad = int((both.groupby("item_id")["prompt_sha256"].nunique() > 1).sum())
        if n_bad:
            warnings.append(
                f"{split}: {n_bad} items were rendered to different input tokens in different "
                "runs (prompt_sha256 differs), so conditions did not see identical prompts."
            )
    settings = [run.manifest.inference for run in runs.values()]
    if len({json.dumps(s, sort_keys=True) for s in settings}) > 1:
        keys = sorted({k for s in settings for k in s})
        differing = [
            k for k in keys if len({json.dumps(s.get(k), sort_keys=True) for s in settings}) > 1
        ]
        warnings.append(
            f"{split}: inference settings differ across eval runs (keys: {', '.join(differing)}). "
            "Spec section 6 requires identical settings for every condition."
        )


def _analyze_split(
    split: str,
    runs: list[tuple[str, str]],
    primary: dict[str, str],
    boot: Bootstrapper,
    warnings: list[str],
    expected_adapters: dict[tuple[str, str], list[str]],
) -> SplitAnalysis:
    items = load_eval_items(split)
    expected = expected_counts(items)
    hashes = current_eval_hashes()
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    loaded: dict[tuple[str, str], Run] = {}
    inputs, missing = [], []
    for condition, label in runs:
        lookup = find_run(
            split, condition, label, hashes, expected_adapters.get((condition, label))
        )
        if lookup.status != "ok":
            missing.append(lookup.describe())
            continue
        run = lookup.run
        df, unmatched = score_run(run, items)
        frames[run.key], loaded[run.key] = df, run
        inputs.append(
            {
                "condition": condition,
                "checkpoint": label,
                "eval_run_id": run.eval_run_id,
                "path": str(run.directory.relative_to(output_root())),
                "results_sha256": run.results_sha256,
                "n_results": len(run.results),
                "n_scored": int(len(df)),
                "n_duplicate_rows": run.n_duplicates,
                "n_unmatched": unmatched,
                "inference": run.manifest.inference,
                "adapter_sha256": run.manifest.adapter_sha256,
            }
        )
        if unmatched:
            warnings.append(
                f"{split}/{condition}/{label}: {unmatched} results have no matching eval item "
                "and were not scored."
            )
        if run.n_duplicates:
            warnings.append(
                f"{split}/{condition}/{label}: {run.n_duplicates} repeated item rows (resumed "
                "run?); the last row per item was used."
            )
    _consistency_warnings(split, frames, loaded, warnings)

    summaries = {key: summarize_run(df, boot, expected) for key, df in frames.items()}
    metrics = [
        {"condition": c, "checkpoint": k, "task": task, **s}
        for (c, k), per_task in summaries.items()
        for task, s in per_task.items()
    ]
    biases, invalid = [], []
    for (condition, label), df in frames.items():
        for task in TASKS:
            sub = df[df["task"] == task]
            if len(sub):
                biases.append(
                    {"condition": condition, "checkpoint": label, "task": task}
                    | bias_summary(sub, boot)
                )
        invalid.append({"condition": condition, "checkpoint": label} | invalid_output_summary(df))
    result = {
        "primary": primary,
        "expected_items": expected,
        "inputs": inputs,
        "missing": missing,
        "metrics": metrics,
        "comparison": _comparison(frames, summaries, primary, boot),
        "composition": _composition(frames, summaries, primary, boot),
        "biases": biases,
        "invalid_outputs": invalid,
    }
    return SplitAnalysis(result, summaries, loaded)


def _expected_adapters(
    selection: SelectionManifest | None,
) -> dict[tuple[str, str], list[str]]:
    """Exact adapter digests each selected condition's eval runs must have loaded."""
    if selection is None:
        return {}
    b, g = selection.belief.adapter_sha256, selection.goal.adapter_sha256
    labels = selected_checkpoints(selection)
    return {
        ("base", "base"): [],
        ("belief", labels["belief"]): [b],
        ("goal", labels["goal"]): [g],
        ("belief+goal", labels["belief+goal"]): [b, g],
    }


def _selection_warnings(
    selection: SelectionManifest, dev: SplitAnalysis, warnings: list[str]
) -> None:
    hashes = current_eval_hashes()
    if any(hashes.get(rel) != h for rel, h in selection.dev_eval_files_sha256.items()):
        warnings.append("Dev eval files changed after the checkpoint selection was frozen.")
    for corpus in ("belief", "goal"):
        choice = getattr(selection, corpus)
        run = dev.runs.get((corpus, choice.checkpoint))
        if run is not None and run.eval_run_id not in choice.eval_run_ids:
            warnings.append(
                f"The dev run now found for {choice.checkpoint} ({run.eval_run_id}) is not the "
                f"one the selection was based on ({', '.join(choice.eval_run_ids)})."
            )


def _cost(cfg: PilotConfig) -> dict:
    caps = {
        "generation": cfg.generation.max_spend_usd,
        "training": cfg.training.max_spend_usd,
        "runpod": cfg.runpod.max_spend_usd,
    }
    path = spend_ledger_path()
    if not path.exists():
        return {"ledger_found": False, "caps_usd": caps, "spent_usd": None}
    spent = {category: round(spend.spent(category, path), 4) for category in caps}
    return {"ledger_found": True, "caps_usd": caps, "spent_usd": spent}


def analyze(cfg: PilotConfig, stage: str | None = None) -> dict:
    """Run one analysis stage, write its outputs, and return the results dict."""
    stage = stage or auto_stage()
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
    if stage == "final":
        selection = load_selection()  # raises SelectionMissingError: no test analysis without it
    elif stage == "dev" and selection_path().exists():
        selection = load_selection()
    else:
        selection = None

    boot = Bootstrapper(cfg.evaluation.bootstrap_samples, cfg.evaluation.bootstrap_seed)
    warnings: list[str] = []
    analyses: dict[str, SplitAnalysis] = {}
    for split, spec in _plan(stage, selection).items():
        if split == "test":
            _guard_test_runs(spec["runs"], selection)
        analyses[split] = _analyze_split(
            split, spec["runs"], spec["primary"], boot, warnings, _expected_adapters(selection)
        )
    if selection is not None:
        _selection_warnings(selection, analyses["dev"], warnings)

    dev_curves = None
    if stage != "baseline":
        dev = analyses["dev"]
        run_ids = {key: run.eval_run_id for key, run in dev.runs.items()}
        dev_curves = build_dev_curves(cfg, dev.summaries, run_ids)
    splits = {name: a.json for name, a in analyses.items()}

    results = {
        "stage": stage,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "headline_split": "test" if stage == "final" else "dev",
        "method": METHOD.format(
            n=cfg.evaluation.bootstrap_samples, seed=cfg.evaluation.bootstrap_seed
        ),
        "settings": {
            "bootstrap_samples": cfg.evaluation.bootstrap_samples,
            "bootstrap_seed": cfg.evaluation.bootstrap_seed,
            "interval": "95% percentile bootstrap over trios",
            "decision_rules": cfg.decision_rules.model_dump(),
        },
        "selection": None
        if selection is None
        else {
            "created_at": selection.created_at,
            "rule": selection.rule,
            "eligibility": selection.eligibility,
            "belief": selection.belief.model_dump(mode="json"),
            "goal": selection.goal.model_dump(mode="json"),
            "composed_checkpoint": composed_label(selection),
        },
        "splits": splits,
        "dev_curves": dev_curves,
        "decision_rules": evaluate_rules(cfg, stage, splits, selection),
        "cost": _cost(cfg),
        "warnings": warnings,
    }
    out = report_dir() / stage
    results["plots"] = plots.make_plots(results, out / "plots")
    write_json(out / "results.json", results)
    (out / "report.md").write_text(report.render(results))
    return results
