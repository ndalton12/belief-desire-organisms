"""Analysis, selection and report tests on synthetic eval items and results.

Every test builds eval files and eval runs in a temporary output folder with known
planted effects, then checks the analysis recovers them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from belief_desire_organisms import metrics, rules
from belief_desire_organisms.analyze import analyze, auto_stage
from belief_desire_organisms.config import load_config
from belief_desire_organisms.metrics import Bootstrapper, trio_means
from belief_desire_organisms.parsing import parse_choice
from belief_desire_organisms.paths import (
    EVAL_DEV_FILE,
    EVAL_DIAGNOSTICS_FILE,
    EVAL_TEST_FILE,
    dataset_dir,
    report_dir,
    results_dir,
    selection_path,
)
from belief_desire_organisms.schemas import (
    FACTS_IN_PROMPT_TASKS,
    LETTERS,
    EvalItem,
    EvalKey,
    EvalResult,
    EvalRunManifest,
    write_json,
    write_jsonl,
)
from belief_desire_organisms.selection import (
    SelectionExistsError,
    SelectionMissingError,
    select_checkpoints,
)

FRUITS = ("apple", "pear", "plum")
PERMS = list(itertools.permutations(range(3)))
TASKS = metrics.TASKS
N_WORDINGS = 5


# ---------------------------------------------------------------------------
# Synthetic world


def _trios(split: str, task: str) -> list[str]:
    prefix = "x" if task == "transfer" else "t"
    start = 1 if split == "dev" else 11
    return [f"{prefix}{i:02d}" for i in range(start, start + 10)]


def make_items(split: str) -> list[EvalItem]:
    items = []
    for task in TASKS:
        for trio in _trios(split, task):
            people = [f"{trio}-{f}" for f in FRUITS]  # index 0 is the apple-liker
            if task == "recall":
                for j, person in enumerate(people):
                    for w in range(2):
                        scen = f"{split}-recall-{person}-w{w}"
                        for p, perm in enumerate(PERMS):
                            fruits = [FRUITS[i] for i in perm]
                            key = EvalKey(
                                target_letter=LETTERS[fruits.index(FRUITS[j])],
                                option_fruits=fruits,
                                option_person_ids=None,
                                subject_person_id=person,
                            )
                            items.append(_item(split, task, scen, trio, f"w{w}", p, key))
                continue
            for w in range(N_WORDINGS):
                scen = f"{split}-{task}-{trio}-w{w}"
                for p, perm in enumerate(PERMS):
                    persons = [people[i] for i in perm]
                    facts = [people[(i + p) % 3] for i in range(3)]
                    key = EvalKey(
                        target_letter=LETTERS[perm.index(0)],
                        option_fruits=[FRUITS[i] for i in perm],
                        option_person_ids=persons,
                        subject_person_id=None,
                        fact_order_person_ids=facts if task in FACTS_IN_PROMPT_TASKS else None,
                    )
                    items.append(_item(split, task, scen, trio, f"w{w}", p, key))
    return items


def _item(split, task, scen, trio, wording, perm, key) -> EvalItem:
    return EvalItem(
        item_id=f"{scen}-p{perm}",
        split=split,
        task=task,
        scenario_id=scen,
        trio_id=trio,
        wording_id=wording,
        permutation=perm,
        prompt=f"prompt for {scen} order {perm}",
        key=key,
    )


@pytest.fixture
def cfg():
    c = load_config()
    return c.model_copy(
        update={"evaluation": c.evaluation.model_copy(update={"bootstrap_samples": 2000})}
    )


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    items = {"dev": make_items("dev"), "test": make_items("test")}
    root = dataset_dir()
    write_jsonl(root / EVAL_DEV_FILE, [i for i in items["dev"] if i.task == "main"])
    write_jsonl(root / EVAL_TEST_FILE, [i for i in items["test"] if i.task == "main"])
    write_jsonl(
        root / EVAL_DIAGNOSTICS_FILE,
        [i for i in items["dev"] + items["test"] if i.task != "main"],
    )
    return items


# ---------------------------------------------------------------------------
# Planting results


Rate = float | Callable[[str], float]


def planted(
    items: list[EvalItem],
    rates: dict[str, Rate],
    invalid: dict[str, float] | None = None,
    default: Rate = 1.0,
) -> dict[str, str]:
    """item_id -> raw output. Within each (task, trio), exactly round(rate * n) items
    answer the target, round(invalid * n) give an unparseable answer, the rest pick a
    non-target letter."""
    invalid = invalid or {}
    groups = defaultdict(list)
    for it in items:
        groups[(it.task, it.trio_id)].append(it)
    out = {}
    for (task, trio), its in groups.items():
        its.sort(key=lambda i: i.item_id)
        r = rates.get(task, default)
        r = r(trio) if callable(r) else r
        n_hit = round(r * len(its))
        n_bad = round(invalid.get(task, 0.0) * len(its))
        for k, it in enumerate(its):
            target = LETTERS.index(it.key.target_letter)
            if k < n_hit:
                out[it.item_id] = it.key.target_letter
            elif k < n_hit + n_bad:
                out[it.item_id] = "I would pick A"
            else:
                out[it.item_id] = LETTERS[(target + 1 + k % 2) % 3]
    return out


def _digests(condition: str, checkpoint: str) -> tuple[dict, dict]:
    if condition == "base":
        return {}, {}
    names = checkpoint.split("+")
    return (
        {n: f"sha-{n}" for n in names},
        {n: f"tinker://run/external_weights/{n}" for n in names},
    )


def write_run(
    split: str,
    condition: str,
    checkpoint: str,
    items: list[EvalItem],
    answers: dict[str, str],
    *,
    run_id: str | None = None,
    eval_hashes: dict[str, str] | None = None,
    logprobs: Callable[[EvalItem], dict] | None = None,
    digests: dict[str, str] | None = None,
) -> None:
    run_id = run_id or f"run-{condition}-{checkpoint}"
    hashes = metrics.current_eval_hashes()
    if eval_hashes is None:
        eval_hashes = {rel: hashes[rel] for rel in metrics.eval_files_for_split(split)}
    default_digests, sources = _digests(condition, checkpoint)
    digests = default_digests if digests is None else digests
    d = results_dir() / split / condition / checkpoint / run_id
    write_json(
        d / "manifest.json",
        EvalRunManifest(
            eval_run_id=run_id,
            split=split,
            condition=condition,
            checkpoint=checkpoint,
            dataset_revision=None,
            eval_files_sha256=eval_hashes,
            base_model="Qwen/Qwen3-8B",
            base_model_revision="rev",
            adapter_sha256=digests,
            adapter_sources=sources,
            inference={"enable_thinking": False, "max_new_tokens": 16, "do_sample": False},
            package_versions={},
        ),
    )
    rows = []
    for it in items:
        if it.item_id not in answers:
            continue
        raw = answers[it.item_id]
        parsed = parse_choice(raw)
        rows.append(
            EvalResult(
                eval_run_id=run_id,
                item_id=it.item_id,
                split=split,
                task=it.task,
                scenario_id=it.scenario_id,
                trio_id=it.trio_id,
                condition=condition,
                checkpoint=checkpoint,
                prompt_sha256=hashlib.sha256(it.prompt.encode()).hexdigest(),
                raw_output=raw,
                parsed_choice=parsed,
                valid=parsed is not None,
                truncated=False,
                letter_logprobs=logprobs(it) if logprobs else None,
                generation_config={"max_new_tokens": 16},
            )
        )
    write_jsonl(d / "results.jsonl", rows)


def by_condition(results: dict, split: str, task: str, condition: str) -> dict:
    return results["splits"][split]["comparison"][task]["by_condition"][condition]


def rule(results: dict, rule_id: str) -> dict:
    return next(r for r in results["decision_rules"] if r["id"] == rule_id)


def alternating(a: float, b: float) -> Callable[[str], float]:
    """a for odd-numbered trios, b for even-numbered ones (mean (a+b)/2 over 10 trios)."""
    return lambda trio: a if int(trio[1:]) % 2 else b


# ---------------------------------------------------------------------------
# Aggregation and bootstrap


def test_invalid_outputs_count_as_failures_unconditionally(world, cfg):
    # Per trio on the main task: 20/30 target, 5/30 invalid, 5/30 other letters.
    answers = planted(world["dev"], {"main": 20 / 30}, invalid={"main": 5 / 30})
    write_run("dev", "base", "base", world["dev"], answers)
    res = analyze(cfg, stage="baseline")
    row = by_condition(res, "dev", "main", "base")
    assert row["target_rate"]["estimate"] == pytest.approx(20 / 30)
    assert row["target_rate_given_valid"]["estimate"] == pytest.approx(20 / 25)
    assert row["valid_rate"]["estimate"] == pytest.approx(25 / 30)
    assert row["n_items"] == row["n_expected"] == 300
    inv = res["splits"]["dev"]["invalid_outputs"][0]
    assert inv["invalid_by_task"] == {"main": 50}
    assert inv["most_common_invalid"] == [["I would pick A", 50]]


def test_aggregation_is_within_scenario_then_within_trio():
    df = pd.DataFrame(
        # trio t1: scenario s1 = 6 hits, scenario s2 = 3 misses (unbalanced on purpose)
        [{"trio_id": "t1", "scenario_id": "s1", "hit": 1.0}] * 6
        + [{"trio_id": "t1", "scenario_id": "s2", "hit": 0.0}] * 3
        # trio t2: one scenario, all misses
        + [{"trio_id": "t2", "scenario_id": "s3", "hit": 0.0}] * 6
    )
    tm = trio_means(df, "hit")
    assert tm.to_dict() == {"t1": 0.5, "t2": 0.0}  # not 6/9 for t1
    est = Bootstrapper(1000, 0).mean(tm)
    assert est.estimate == pytest.approx(0.25)  # mean over trios, not 6/15 over items
    assert est.n_trios == 2


def test_bootstrap_is_paired_and_deterministic():
    trios = [f"t{i:02d}" for i in range(10)]
    base = pd.Series(np.linspace(0.0, 0.9, 10), index=trios)
    combined = base + 0.2
    boot = Bootstrapper(5000, 7)
    diff = boot.linear([combined, base], [1, -1])
    # Every trio gains exactly 0.2, so the paired interval collapses on it, although
    # each condition on its own varies a lot between trios.
    assert diff.estimate == pytest.approx(0.2)
    assert diff.ci_low == pytest.approx(0.2) and diff.ci_high == pytest.approx(0.2)
    marginal = boot.mean(combined)
    assert marginal.ci_high - marginal.ci_low > 0.3
    # Deterministic for a fixed seed, different for another seed.
    assert Bootstrapper(5000, 7).mean(combined) == marginal
    assert Bootstrapper(5000, 8).mean(combined) != marginal


def test_composition_contrasts_paired_over_trios_and_interaction(world, cfg):
    """Planted main-task rates (trio offset o = -0.1 / +0.1, gain g = 0.4 / 0.6):
    M0 0.1+o, B 0.2+o, D 0.3+o, B+D = B + g. Means: 0.1, 0.2, 0.3, 0.7."""
    o = alternating(-0.1, 0.1)
    g = lambda t: 0.4 if int(t[1:]) <= 5 or int(t[1:]) > 15 else 0.6  # noqa: E731
    main_rates = {
        "base": lambda t: 0.1 + o(t),
        "belief": lambda t: 0.2 + o(t),
        "goal": lambda t: 0.3 + o(t),
        "belief+goal": lambda t: 0.2 + o(t) + g(t),
    }
    labels = {"belief": "belief-5000000", "goal": "goal-3000000"}
    labels["belief+goal"] = "belief-5000000+goal-3000000"
    labels["base"] = "base"
    for split in ("dev", "test"):
        for cond, label in labels.items():
            write_run(
                split, cond, label, world[split], planted(world[split], {"main": main_rates[cond]})
            )
    select_checkpoints(cfg)
    res = analyze(cfg, stage="final")
    comp = res["splits"]["test"]["composition"]
    assert comp["available"]
    planted_effects = {
        "bd_minus_belief": 0.5,
        "bd_minus_goal": 0.4,
        "bd_minus_base": 0.6,
        "interaction": 0.3,  # (B + g) - B - D + M0 = g - 0.2
    }
    for name, value in planted_effects.items():
        e = comp[name]
        assert e["estimate"] == pytest.approx(value, abs=1e-6), name
        assert e["ci_low"] <= value <= e["ci_high"], name
        assert e["ci_high"] - e["ci_low"] > 0.05, name  # g varies by trio: a real interval
        assert e["n_trios"] == 10
    # The trio offset o is shared by all conditions, so it cancels in paired differences:
    # the paired B+D minus M0 interval is narrower than B+D's own interval.
    bd = by_condition(res, "test", "main", "belief+goal")["target_rate"]
    assert comp["bd_minus_base"]["ci_high"] - comp["bd_minus_base"]["ci_low"] < (
        bd["ci_high"] - bd["ci_low"]
    )
    assert rule(res, "composition")["status"] == "pass"


# ---------------------------------------------------------------------------
# Biases


def test_position_bias_detected(world, cfg):
    answers = planted(world["dev"], {})
    for it in world["dev"]:
        if it.task == "main":
            answers[it.item_id] = "A"  # always the first option
        elif it.task == "goal_expression":  # exactly balanced letters
            answers[it.item_id] = LETTERS[(it.permutation + int(it.wording_id[1:])) % 3]
    write_run("dev", "base", "base", world["dev"], answers)
    res = analyze(cfg, stage="baseline")
    biases = {b["task"]: b for b in res["splits"]["dev"]["biases"]}
    assert biases["main"]["letter_share"]["A"]["estimate"] == pytest.approx(1.0)
    assert biases["main"]["position_bias_flagged"]
    assert not biases["goal_expression"]["position_bias_flagged"]
    for letter in LETTERS:
        share = biases["goal_expression"]["letter_share"][letter]["estimate"]
        assert share == pytest.approx(1 / 3)
    # Always "A" picks each person exactly when they are listed first: 2 of 6 orders.
    rates = [p["rate"] for p in biases["main"]["persons"]]
    assert len(rates) == 30 and rates == pytest.approx([1 / 3] * 30, abs=1e-6)
    assert "flagged" in (report_dir() / "baseline" / "report.md").read_text()


def test_fruit_and_wording_effects(world, cfg):
    # Apple-liker chosen in every item of wording w0 and never otherwise (main task).
    answers = planted(world["dev"], {})
    for it in world["dev"]:
        if it.task == "main":
            target = LETTERS.index(it.key.target_letter)
            hit = it.wording_id == "w0"
            answers[it.item_id] = LETTERS[target if hit else (target + 1) % 3]
    write_run("dev", "base", "base", world["dev"], answers)
    res = analyze(cfg, stage="baseline")
    main = next(b for b in res["splits"]["dev"]["biases"] if b["task"] == "main")
    assert main["wording_target_rate"]["w0"]["estimate"] == pytest.approx(1.0)
    assert main["wording_target_rate"]["w1"]["estimate"] == pytest.approx(0.0)
    assert main["fruit_share"]["apple"]["estimate"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Decision rules


@pytest.mark.parametrize("n_correct,status", [(27, "pass"), (26, "fail")])
def test_capability_gate_from_baseline_results_alone(world, cfg, n_correct, status):
    answers = planted(world["dev"], {"capability_ceiling": n_correct / 30})
    write_run("dev", "base", "base", world["dev"], answers)
    assert auto_stage() == "baseline"
    res = analyze(cfg)  # no trained checkpoints, no selection: baseline stage
    assert res["stage"] == "baseline"
    gate = rule(res, "capability_ceiling")
    assert gate["status"] == status
    assert gate["values"]["capability_ceiling"]["estimate"] == pytest.approx(n_correct / 30)
    others = [r["status"] for r in res["decision_rules"] if r["id"] != "capability_ceiling"]
    assert set(others) == {"not_evaluated"}


def _split(rows: dict, vs_base: dict | None = None, composition: dict | None = None) -> dict:
    """Minimal split result: rows = {(task, condition): target rate}."""
    comparison: dict = {}
    for (task, condition), rate in rows.items():
        entry = comparison.setdefault(task, {"by_condition": {}, "vs_base": {}, "vs_chance": {}})
        entry["by_condition"][condition] = {
            "checkpoint": condition,
            "n_items": 300,
            "n_expected": 300,
            "complete": True,
            "target_rate": {"estimate": rate, "ci_low": rate - 0.05, "ci_high": rate + 0.05},
            "valid_rate": {"estimate": 1.0, "ci_low": 1.0, "ci_high": 1.0},
        }
    for (task, condition), diff in (vs_base or {}).items():
        comparison[task]["vs_base"][condition] = {
            "estimate": diff,
            "ci_low": diff - 0.05,
            "ci_high": diff + 0.05,
        }
    return {"comparison": comparison, "composition": composition or {"available": False}}


class _Choice:
    def __init__(self, checkpoint):
        self.checkpoint, self.qualified = checkpoint, True


class _Selection:
    belief, goal = _Choice("belief-1"), _Choice("goal-1")


@pytest.mark.parametrize(
    "recall,gain,status",
    [(0.80, 0.20, "pass"), (0.7999, 0.30, "fail"), (0.95, 0.1999, "fail")],
)
def test_belief_recall_rule_edges(cfg, recall, gain, status):
    dev = _split(
        {("recall", "belief"): recall, ("recall", "base"): recall - gain},
        vs_base={("recall", "belief"): gain},
    )
    out = rules.evaluate_rules(cfg, "dev", {"dev": dev}, _Selection())
    assert next(r for r in out if r["id"] == "belief_recall")["status"] == status


@pytest.mark.parametrize(
    "vs_b,vs_d,valid,status",
    [
        (0.15, 0.15, 0.95, "pass"),
        (0.1499, 0.30, 0.99, "fail"),
        (0.30, 0.1499, 0.99, "fail"),
        (0.30, 0.30, 0.9499, "fail"),
    ],
)
def test_composition_rule_edges(cfg, vs_b, vs_d, valid, status):
    def e(x):
        return {"estimate": x, "ci_low": x - 0.1, "ci_high": x + 0.1}

    test = _split(
        {},
        composition={
            "available": True,
            "complete": True,
            "bd_minus_belief": e(vs_b),
            "bd_minus_goal": e(vs_d),
            "bd_valid_rate": e(valid),
        },
    )
    out = rules.evaluate_rules(cfg, "final", {"dev": _split({}), "test": test}, _Selection())
    assert next(r for r in out if r["id"] == "composition")["status"] == status


@pytest.mark.parametrize("diff,status", [(0.10, "pass"), (0.1001, "flagged")])
def test_separation_flag_edges(cfg, diff, status):
    test = _split(
        {("goal_expression", "belief"): 0.5, ("goal_expression", "base"): 0.5 - diff},
        vs_base={("goal_expression", "belief"): diff},
    )
    out = rules.evaluate_rules(cfg, "final", {"dev": _split({}), "test": test}, _Selection())
    sep = next(r for r in out if r["id"] == "separation_belief")
    assert sep["status"] == status and sep["split"] == "test"


# ---------------------------------------------------------------------------
# Checkpoint selection


def _plant_dev_checkpoints(world):
    """Dev results where the right picks are belief-3000000 and goal-3000000."""
    dev = world["dev"]
    write_run("dev", "base", "base", dev, planted(dev, {"recall": 1 / 3, "goal_expression": 1 / 3}))
    belief = {
        "belief-0": (1.0, {}),  # untrained pipeline check: never a candidate
        "belief-1000000": (0.5, {}),
        "belief-3000000": (11 / 12, {}),  # recall has 36 items per trio: use twelfths
        "belief-5000000": (10 / 12, {}),
        # Best recall but 10% invalid on the main task: not eligible.
        "belief-10000000": (1.0, {"main": 0.1}),
    }
    for label, (recall, invalid) in belief.items():
        rates = {"recall": recall, "goal_expression": 1 / 3, "main": 1 / 3}
        write_run("dev", "belief", label, dev, planted(dev, rates, invalid=invalid))
    goal = {"goal-1000000": 0.6, "goal-3000000": 0.8, "goal-5000000": 0.8}  # tie: fewer tokens
    for label, rate in goal.items():
        rates = {"goal_expression": rate, "recall": 1 / 3, "main": 1 / 3}
        write_run("dev", "goal", label, dev, planted(dev, rates))


def test_selection_uses_dev_only(world, cfg, monkeypatch):
    _plant_dev_checkpoints(world)
    test = world["test"]
    # Misleading test results: here belief-5000000 and goal-5000000 look far better.
    for label, recall in (("belief-3000000", 0.1), ("belief-5000000", 1.0)):
        write_run("test", "belief", label, test, planted(test, {"recall": recall}))
    for label, rate in (("goal-3000000", 0.1), ("goal-5000000", 1.0)):
        write_run("test", "goal", label, test, planted(test, {"goal_expression": rate}))
    broken = results_dir() / "test" / "belief" / "belief-3000000" / "run-bad"
    broken.mkdir(parents=True)
    (broken / "results.jsonl").write_text("not json\n")

    reads = []
    real_read = metrics.read_result_file
    monkeypatch.setattr(metrics, "read_result_file", lambda p: reads.append(p) or real_read(p))
    sel = select_checkpoints(cfg)

    assert reads and all(p.relative_to(results_dir()).parts[0] == "dev" for p in reads)
    assert sel.belief.checkpoint == "belief-3000000" and sel.belief.qualified
    assert sel.goal.checkpoint == "goal-3000000" and sel.goal.qualified
    assert sel.belief.value == pytest.approx(11 / 12, abs=1e-6)
    assert sel.belief.adapter_sha256 == "sha-belief-3000000"
    assert sel.belief.tinker_path == "tinker://run/external_weights/belief-3000000"
    assert sel.belief.eval_run_ids == ["run-belief-belief-3000000"]
    assert set(sel.dev_eval_files_sha256) == {EVAL_DEV_FILE, EVAL_DIAGNOSTICS_FILE}

    saved = json.loads(selection_path().read_text())
    curve = {e["checkpoint"]: e for e in saved["dev_curves"]["belief"]}
    assert list(curve) == [
        "belief-0",
        "belief-1000000",
        "belief-3000000",
        "belief-5000000",
        "belief-10000000",
    ]  # every checkpoint, ordered by processed tokens
    assert not curve["belief-0"]["candidate"]
    assert not curve["belief-10000000"]["eligible"]
    assert any("valid-output" in r for r in curve["belief-10000000"]["ineligible_reasons"])


def test_selection_refuses_to_overwrite(world, cfg):
    _plant_dev_checkpoints(world)
    first = select_checkpoints(cfg)
    with pytest.raises(SelectionExistsError):
        select_checkpoints(cfg)
    assert json.loads(selection_path().read_text())["created_at"] == first.created_at
    assert select_checkpoints(cfg, force=True).belief.checkpoint == "belief-3000000"


def test_selection_falls_back_when_nothing_is_eligible(world, cfg):
    dev = world["dev"]
    write_run("dev", "base", "base", dev, planted(dev, {}))
    for label, recall in (("belief-1000000", 0.4), ("belief-3000000", 0.6)):
        rates = {"recall": recall, "capability_ceiling": 0.5}  # capability damaged
        write_run("dev", "belief", label, dev, planted(dev, rates))
    write_run("dev", "goal", "goal-1000000", dev, planted(dev, {"goal_expression": 0.7}))
    sel = select_checkpoints(cfg)
    assert sel.belief.checkpoint == "belief-3000000" and not sel.belief.qualified
    assert sel.goal.qualified


def test_final_analysis_refuses_without_selection(world, cfg):
    _plant_dev_checkpoints(world)
    write_run("test", "base", "base", world["test"], planted(world["test"], {}))
    with pytest.raises(SelectionMissingError):
        analyze(cfg, stage="final")
    assert not (report_dir() / "final").exists()


def test_final_analysis_reads_only_selected_test_checkpoints(world, cfg, monkeypatch):
    _plant_dev_checkpoints(world)
    sel = select_checkpoints(cfg)
    test = world["test"]
    selected = {
        "base": "base",
        "belief": sel.belief.checkpoint,
        "goal": sel.goal.checkpoint,
        "belief+goal": f"{sel.belief.checkpoint}+{sel.goal.checkpoint}",
    }
    for cond, label in selected.items():
        write_run("test", cond, label, test, planted(test, {"main": 0.5}))
    write_run("test", "belief", "belief-5000000", test, planted(test, {"main": 1.0}))

    reads = []
    real_read = metrics.read_result_file
    monkeypatch.setattr(metrics, "read_result_file", lambda p: reads.append(p) or real_read(p))
    res = analyze(cfg)  # a frozen selection exists: final stage
    assert res["stage"] == "final"
    rel = [p.relative_to(results_dir()).parts for p in reads]
    test_reads = {parts[1:3] for parts in rel if parts[0] == "test"}
    assert test_reads == set(selected.items())
    used = {(i["condition"], i["checkpoint"]) for i in res["splits"]["test"]["inputs"]}
    assert used == set(selected.items())
    assert by_condition(res, "test", "main", "belief")["target_rate"]["estimate"] == 0.5


# ---------------------------------------------------------------------------
# Report, missing data, run verification


def test_report_states_missing_conditions(world, cfg):
    _plant_dev_checkpoints(world)
    sel = select_checkpoints(cfg)
    test = world["test"]
    write_run("test", "base", "base", test, planted(test, {"main": 0.3}))
    write_run("test", "belief", sel.belief.checkpoint, test, planted(test, {"main": 0.4}))
    res = analyze(cfg, stage="final")
    md = (report_dir() / "final" / "report.md").read_text()
    missing = {m["condition"] for m in res["splits"]["test"]["missing"]}
    assert missing == {"goal", "belief+goal"}
    assert "Missing results" in md
    assert f"test / goal / {sel.goal.checkpoint}: missing" in md
    assert "Composition differences not computed" in md
    assert rule(res, "composition")["status"] == "not_evaluated"
    assert "| D (goal) | goal-3000000 | missing |" in md
    assert "No spend ledger found" in md
    assert "No test-split eval run recorded letter log-probabilities." in md
    for name in ("conditions", "dev_curves_belief", "dev_curves_goal", "position_bias"):
        assert (report_dir() / "final" / res["plots"][name]).exists()
    assert json.loads((report_dir() / "final" / "results.json").read_text())["stage"] == "final"


def test_runs_must_match_current_eval_files(world, cfg):
    dev = world["dev"]
    write_run("dev", "base", "base", dev, planted(dev, {}), eval_hashes={EVAL_DEV_FILE: "stale"})
    res = analyze(cfg, stage="baseline")
    (m,) = res["splits"]["dev"]["missing"]
    assert m["status"] == "unverified" and "changed since this run" in m["detail"]
    assert rule(res, "capability_ceiling")["status"] == "not_evaluated"

    write_run("dev", "base", "base", dev, planted(dev, {}), run_id="run-a")
    write_run("dev", "base", "base", dev, planted(dev, {}), run_id="run-b")
    res = analyze(cfg, stage="baseline")
    assert res["splits"]["dev"]["missing"][0]["status"] == "ambiguous"


def test_dev_stage_curves_and_selection_preview(world, cfg):
    _plant_dev_checkpoints(world)
    assert auto_stage() == "dev"
    res = analyze(cfg)
    curves = res["dev_curves"]
    assert [e["processed_tokens"] for e in curves["goal"]] == [1_000_000, 3_000_000, 5_000_000]
    assert curves["belief_preview"] == {"checkpoint": "belief-3000000", "qualified": True}
    recall = [e["criteria"]["recall"] for e in curves["belief"]]
    assert recall == pytest.approx([1.0, 0.5, 11 / 12, 10 / 12, 1.0], abs=1e-6)
    md = (report_dir() / "dev" / "report.md").read_text()
    assert "would pick **belief-3000000**" in md
    assert rule(res, "belief_recall")["status"] == "not_evaluated"


def test_analysis_is_deterministic(world, cfg):
    o = alternating(0.2, 0.6)
    write_run("dev", "base", "base", world["dev"], planted(world["dev"], {"main": o}))
    first = analyze(cfg, stage="baseline")
    second = analyze(cfg, stage="baseline")
    first.pop("generated_at"), second.pop("generated_at")
    assert first == second
    e = by_condition(first, "dev", "main", "base")["target_rate"]
    assert e["ci_low"] < 0.4 < e["ci_high"]


def test_continuous_readout_from_letter_logprobs(world, cfg):
    def lp(it):
        return {
            letter: math.log(0.7 if letter == it.key.target_letter else 0.15) for letter in LETTERS
        }

    dev = world["dev"]
    write_run("dev", "base", "base", dev, planted(dev, {}), logprobs=lp)
    res = analyze(cfg, stage="baseline")
    row = by_condition(res, "dev", "main", "base")
    assert row["target_prob"]["estimate"] == pytest.approx(0.7)
    assert row["target_prob_coverage"] == 1.0
    assert "Normalised target probability" in (report_dir() / "baseline" / "report.md").read_text()


def test_corrupt_runs_are_reported_not_fatal(world, cfg):
    dev = world["dev"]
    write_run("dev", "base", "base", dev, planted(dev, {}))
    path = results_dir() / "dev" / "base" / "base" / "run-base-base" / "results.jsonl"
    path.write_text(path.read_text()[:-200])  # killed mid-write: truncated last line
    res = analyze(cfg, stage="baseline")
    (m,) = res["splits"]["dev"]["missing"]
    assert m["status"] == "unreadable"
    assert rule(res, "capability_ceiling")["status"] == "not_evaluated"
    assert "unreadable" in (report_dir() / "baseline" / "report.md").read_text()

    manifest = path.parent / "manifest.json"
    write_run("dev", "base", "base", dev, planted(dev, {}))
    manifest.write_text("{")
    res = analyze(cfg, stage="baseline")
    assert res["splits"]["dev"]["missing"][0]["status"] == "unverified"
    assert "unreadable manifest.json" in res["splits"]["dev"]["missing"][0]["detail"]


# ---------------------------------------------------------------------------
# Incomplete coverage, required hashes, frozen adapter identities


def test_partial_runs_cannot_pass_gates(world, cfg):
    dev = world["dev"]
    answers = planted(dev, {"capability_ceiling": 0.0})
    keep = next(i for i in dev if i.task == "capability_ceiling")
    answers[keep.item_id] = keep.key.target_letter  # the one capability row is correct
    partial = {k: v for k, v in answers.items() if not k.startswith("dev-capability")}
    partial[keep.item_id] = answers[keep.item_id]
    write_run("dev", "base", "base", dev, partial)
    res = analyze(cfg, stage="baseline")
    row = by_condition(res, "dev", "capability_ceiling", "base")
    assert row["target_rate"]["estimate"] == 1.0 and not row["complete"]
    gate = rule(res, "capability_ceiling")
    assert gate["status"] == "not_evaluated" and "1 of 300 items" in gate["note"]


def test_paired_contrasts_use_only_common_items(world, cfg):
    def frame(wordings, hit):
        return pd.DataFrame(
            [
                {
                    "task": "main",
                    "trio_id": "t1",
                    "scenario_id": f"s{w}",
                    "item_id": f"i{w}{p}",
                    "hit": hit,
                }
                for w in wordings
                for p in range(6)
            ]
        )

    # Same trio, disjoint items: there is nothing to pair.
    series, n = metrics.paired_trio_means([frame("01", 1.0), frame("23", 0.0)], "main")
    assert n == 0 and Bootstrapper(100, 0).linear(series, [1, -1]).estimate is None
    series, n = metrics.paired_trio_means([frame("012", 1.0), frame("234", 0.0)], "main")
    assert n == 6 and [s.to_dict() for s in series] == [{"t1": 1.0}, {"t1": 0.0}]

    # End to end: test runs that share trios but answered different wordings.
    _plant_dev_checkpoints(world)
    sel = select_checkpoints(cfg)
    test = world["test"]
    labels = {
        "base": "base",
        "belief": sel.belief.checkpoint,
        "goal": sel.goal.checkpoint,
        "belief+goal": f"{sel.belief.checkpoint}+{sel.goal.checkpoint}",
    }
    for cond, label in labels.items():
        answers = planted(test, {"main": 0.5})
        if cond in ("base", "belief+goal"):
            keep = "012" if cond == "base" else "234"
            answers = {k: v for k, v in answers.items() if "-main-" not in k or k[-4] in keep}
        write_run("test", cond, label, test, answers)
    res = analyze(cfg, stage="final")
    comp = res["splits"]["test"]["composition"]
    assert comp["available"] and not comp["complete"]
    assert comp["bd_minus_base"]["n_common_items"] == 60  # wording w2 only
    assert comp["bd_minus_belief"]["n_common_items"] == 180
    assert rule(res, "composition")["status"] == "not_evaluated"
    assert "incomplete" in rule(res, "composition")["note"]
    assert "**Incomplete results:**" in (report_dir() / "final" / "report.md").read_text()


def _plant_dev_checkpoints_only_goal(world):
    dev = world["dev"]
    write_run("dev", "goal", "goal-1000000", dev, planted(dev, {"goal_expression": 0.8}))


def test_partial_checkpoint_is_never_selected(world, cfg):
    dev = world["dev"]
    write_run("dev", "base", "base", dev, planted(dev, {}))
    _plant_dev_checkpoints_only_goal(world)
    write_run("dev", "belief", "belief-1000000", dev, planted(dev, {"recall": 0.5}))
    one_recall = {
        k: v
        for k, v in planted(dev, {"recall": 1.0}).items()
        if "-recall-" not in k or k.endswith("-t01-apple-w0-p0")
    }
    write_run("dev", "belief", "belief-3000000", dev, one_recall)
    sel = select_checkpoints(cfg)
    assert sel.belief.checkpoint == "belief-1000000"
    curve = {e["checkpoint"]: e for e in sel.dev_curves["belief"]}
    assert not curve["belief-3000000"]["candidate"]
    assert any("incomplete" in r for r in curve["belief-3000000"]["ineligible_reasons"])


def test_runs_must_record_every_eval_file_of_their_split(world, cfg):
    dev = world["dev"]
    hashes = metrics.current_eval_hashes()
    write_run(
        "dev",
        "base",
        "base",
        dev,
        planted(dev, {}),
        eval_hashes={EVAL_TEST_FILE: hashes[EVAL_TEST_FILE]},
    )
    res = analyze(cfg, stage="baseline")
    (m,) = res["splits"]["dev"]["missing"]
    assert m["status"] == "unverified"
    assert f"does not record the hash of {EVAL_DEV_FILE}" in m["detail"]
    assert f"does not record the hash of {EVAL_DIAGNOSTICS_FILE}" in m["detail"]


def test_base_run_must_not_load_adapters(world, cfg):
    dev = world["dev"]
    write_run("dev", "base", "base", dev, planted(dev, {}), digests={"x": "sha-x"})
    res = analyze(cfg, stage="baseline")
    assert (
        "manifest records 1 adapters; base loads 0" in res["splits"]["dev"]["missing"][0]["detail"]
    )


def test_test_runs_with_substituted_weights_are_not_read(world, cfg, monkeypatch):
    _plant_dev_checkpoints(world)
    sel = select_checkpoints(cfg)
    test = world["test"]
    labels = {
        "base": "base",
        "belief": sel.belief.checkpoint,
        "goal": sel.goal.checkpoint,
        "belief+goal": f"{sel.belief.checkpoint}+{sel.goal.checkpoint}",
    }
    for cond, label in labels.items():
        digests = None
        if cond == "belief":
            digests = {label: "sha-some-other-weights"}
        if cond == "belief+goal":  # selected goal plus an extra, unselected adapter
            digests = {"a": sel.goal.adapter_sha256, "b": "sha-extra"}
        write_run("test", cond, label, test, planted(test, {"main": 1.0}), digests=digests)
    reads = []
    real_read = metrics.read_result_file
    monkeypatch.setattr(metrics, "read_result_file", lambda p: reads.append(p) or real_read(p))
    res = analyze(cfg, stage="final")
    read_test = {
        p.relative_to(results_dir()).parts[1]
        for p in reads
        if p.relative_to(results_dir()).parts[0] == "test"
    }
    assert read_test == {"base", "goal"}
    missing = {m["condition"]: m for m in res["splits"]["test"]["missing"]}
    assert set(missing) == {"belief", "belief+goal"}
    assert "differ from the frozen checkpoint selection" in missing["belief"]["detail"]
    assert rule(res, "composition")["status"] == "not_evaluated"
