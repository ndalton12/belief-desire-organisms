"""The grafting experiment's harness, end to end on tiny models: four arms, dev sweep and
checkpoint choice, the nine test conditions, visible reasoning, and the summary."""

from __future__ import annotations

import json

import pytest
from gpu_eval_helpers import make_item, register_exported_adapter, save_adapter, tiny_model

from belief_desire_organisms import evaluate as ev
from belief_desire_organisms import grafting as gr
from belief_desire_organisms.build_evals import goal_sentence
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import (
    EVAL_DEV_FILE,
    EVAL_DIAGNOSTICS_FILE,
    EVAL_TEST_FILE,
    dataset_dir,
    output_root,
    training_config,
    variants,
)
from belief_desire_organisms.schemas import EvalResult, read_jsonl, write_jsonl
from belief_desire_organisms.tokenizer import get_tokenizer


@pytest.fixture(autouse=True)
def _outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.delenv("BDO_EVAL_MODE", raising=False)
    monkeypatch.delenv("BDO_BELIEF_VARIANT", raising=False)
    monkeypatch.delenv("BDO_GOAL_VARIANT", raising=False)


def _cfg():
    cfg = load_config()
    t = cfg.training.model_copy(
        update={"graft_base_model": "Qwen/Qwen3-8B-Base", "graft_base_model_revision": "1" * 40}
    )
    return cfg.model_copy(update={"training": t})


GOAL = goal_sentence("apple")


def _write_eval_files():
    items = []
    for split in ("dev", "test"):
        items += [make_item(i, split=split) for i in range(6)]
        items += [
            make_item(i, split=split, task="belief_use", prompt=f"{GOAL}\n\nWho gets it? {i}")
            for i in range(6)
        ]
        for task in ("recall", "goal_expression", "capability_ceiling"):
            items += [make_item(i, split=split, task=task) for i in range(3)]
    names, fruits = ["Mira Venn", "Leon Daro", "Sana Pell"], ["apples", "pears", "plums"]
    items = [
        i.model_copy(
            update={
                "key": i.key.model_copy(
                    update={"option_answers": fruits if i.task == "recall" else names}
                )
            }
        )
        for i in items
    ]
    write_jsonl(
        dataset_dir() / EVAL_DEV_FILE, [i for i in items if (i.split, i.task) == ("dev", "main")]
    )
    write_jsonl(
        dataset_dir() / EVAL_TEST_FILE, [i for i in items if (i.split, i.task) == ("test", "main")]
    )
    write_jsonl(dataset_dir() / EVAL_DIAGNOSTICS_FILE, [i for i in items if i.task != "main"])
    (dataset_dir() / ev.DATASET_REVISION_FILE).write_text("rev\n")
    return items


def _register_arms(cfg, tmp_path):
    seed = 0
    for arm in gr.arms():
        names = (
            sorted({"belief-1000000", gr.fixed_checkpoints()[Corpus.BELIEF]})
            if arm.corpus == Corpus.BELIEF
            else ["goal-500000"]
        )
        for name in names:
            seed += 1
            with variants(**{arm.corpus.value: arm.variant}):
                tcfg = training_config(cfg, arm.corpus)
                src = save_adapter(tmp_path / f"src-{arm.key}-{name}", r=4, alpha=8, seed=seed)
                register_exported_adapter(tcfg, arm.corpus, name, src)


def test_other_fruit_items_retarget_the_goal():
    items = _write_eval_files()
    use = [i for i in items if i.task == "belief_use" and i.split == "dev"]
    other = gr.other_fruit_items(use)
    assert len(other) == 2 * len(use)
    pear = other[0]
    assert goal_sentence("pear") in pear.prompt and GOAL not in pear.prompt
    assert pear.key.option_fruits[["A", "B", "C"].index(pear.key.target_letter)] == "pear"
    assert pear.task == gr.OTHER_FRUIT_TASK and pear.item_id.endswith("-gpear")
    assert len({i.item_id for i in other}) == len(other)


def test_adapter_names_and_the_nine_conditions():
    runs = gr.nine_runs()
    labels = [r.label for r in runs]
    assert labels[0] == "base" and len(set(labels)) == 9
    assert "belief_graft-2000000+goal_native-500000" in labels
    assert {r.condition.value for r in runs} == {"base", "belief", "goal", "belief+goal"}


def test_dev_sweep_choice_test_conditions_reasoning_and_summary(tmp_path, monkeypatch):
    cfg = _cfg()
    _write_eval_files()
    _register_arms(cfg, tmp_path)
    monkeypatch.setattr(gr, "load_chat_model", lambda cfg: tiny_model())
    monkeypatch.setitem(ev.REASONING, "max_reasoning_tokens", 3)
    assert get_tokenizer(cfg) is not None

    dev = gr.evaluate_split(cfg, "dev")
    assert len(dev) == 1 + 2 + 2 + 1 + 1  # base + every checkpoint of every arm
    pre = json.loads((output_root() / "results" / gr.PREFLIGHT_FILE).read_text())
    assert pre["base"]["padded_decoding_token_agreement"] == 1.0
    sel = json.loads((output_root() / "results" / gr.SELECTION_FILE).read_text())
    assert set(sel["chosen"]) == {a.key for a in gr.arms()}
    belief_dev = [d for d in dev if d["checkpoint"] == "belief_native-2000000"][0]
    assert belief_dev["rows_by_task"] == {
        "belief_use": 6,
        gr.OTHER_FRUIT_TASK: 12,
        "capability_ceiling": 3,
        "recall": 3,
    }

    test = gr.evaluate_split(cfg, "test")
    labels = {d["checkpoint"] for d in test}
    assert {r.label for r in gr.nine_runs()} <= labels
    pair = [d for d in test if d["checkpoint"] == "belief_graft-2000000+goal_graft-500000"][0]
    info = json.loads(
        (tmp_path / "x").parent.joinpath(pair["run_dir"], ev.RUN_INFO_FILE).read_text()
    )
    checks = info["sessions"][-1]["checks"]
    assert checks["composition"]["passed"] and checks["cat_cross_check"]["passed"]
    manifest = json.loads((tmp_path / pair["run_dir"] / ev.MANIFEST_FILE).read_text())
    assert set(manifest["adapter_sha256"]) == {"belief_graft-2000000", "goal_graft-500000"}
    assert any(k.startswith("derived/") for k in manifest["eval_files_sha256"])
    # graft and native adapters are different files with different digests
    native = [d for d in test if d["checkpoint"] == "belief_native-2000000+goal_graft-500000"][0]
    m2 = json.loads((tmp_path / native["run_dir"] / ev.MANIFEST_FILE).read_text())
    assert (
        m2["adapter_sha256"]["goal_graft-500000"] == manifest["adapter_sha256"]["goal_graft-500000"]
    )
    assert (
        m2["adapter_sha256"]["belief_native-2000000"]
        != manifest["adapter_sha256"]["belief_graft-2000000"]
    )

    monkeypatch.setenv("BDO_EVAL_MODE", "reasoning")
    reasoning = gr.evaluate_split(cfg, "test")
    assert len(reasoning) == 9
    base = [d for d in reasoning if d["checkpoint"] == "base"][0]
    assert set(base["rows_by_task"]) == {"main", "belief_use", gr.OTHER_FRUIT_TASK}
    goal = [d for d in reasoning if d["checkpoint"] == "goal_native-500000"][0]
    assert set(goal["rows_by_task"]) == {"main"}
    rows = list(read_jsonl(tmp_path / goal["run_dir"] / ev.RESULTS_FILE, EvalResult))
    assert all(r.raw_output.startswith(ev.REASONING["opener"]) for r in rows)
    monkeypatch.delenv("BDO_EVAL_MODE")

    summary = gr.summarize(cfg)
    assert set(summary["modes"]) == {"normal", "reasoning"}
    c = summary["modes"]["normal"]["contrasts"]
    assert "origin_interaction:main" in c and "graft_minus_native_belief:belief_use_all" in c
    assert c["origin_interaction:main"]["n_trios"] == 3
    rates = summary["modes"]["normal"]["rates"]["belief_graft-2000000"]
    assert rates["belief_use_all"]["n"] == 18
    assert summary["modes"]["normal"]["complete"] and summary["modes"]["reasoning"]["complete"]
    full = c["belief_graft-2000000+goal_graft-500000:interaction"]["n_common_items"]
    assert full == 6

    # An interrupted run: contrasts pair only the items every condition answered, and the
    # summary says it is incomplete.
    path = tmp_path / pair["run_dir"] / ev.RESULTS_FILE
    lines = path.read_text().splitlines()
    main_rows = [i for i, line in enumerate(lines) if '"task":"main"' in line]
    path.write_text("\n".join(x for i, x in enumerate(lines) if i != main_rows[0]) + "\n")
    partial = gr.summarize(cfg)["modes"]["normal"]
    assert not partial["complete"]
    assert (
        partial["contrasts"]["belief_graft-2000000+goal_graft-500000:interaction"]["n_common_items"]
        == full - 1
    )

    # Results made from other evaluation items are refused.
    mpath = tmp_path / pair["run_dir"] / ev.MANIFEST_FILE
    m = json.loads(mpath.read_text())
    m["eval_files_sha256"] = {k: "0" * 64 for k in m["eval_files_sha256"]}
    mpath.write_text(json.dumps(m))
    with pytest.raises(ev.EvaluationError, match="other evaluation items"):
        gr.summarize(cfg)


def test_reasoning_dev_and_unknown_modes_are_refused(monkeypatch):
    monkeypatch.setenv("BDO_EVAL_MODE", "reasoning")
    with pytest.raises(ev.EvaluationError, match="dev sweep"):
        gr.evaluate_split(_cfg(), "dev")
    monkeypatch.setenv("BDO_EVAL_MODE", "thinking")
    with pytest.raises(ev.EvaluationError, match="not part"):
        gr.evaluate_split(_cfg(), "test")


def test_qa_belief_data_uses_its_own_arms_checkpoints_and_results(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_GRAFT_BELIEF", "qa")
    cfg = _cfg()
    _write_eval_files()
    _register_arms(cfg, tmp_path)
    labels = [r.label for r in gr.nine_runs()]
    assert "belief_qa_graft-500000+goal_native-500000" in labels
    assert not any(lab.startswith("belief_native") for lab in labels)
    assert gr.results_root().name == "results-graft-qa"
    assert gr.results_root("reasoning").name == "results-graft-qa-reasoning"
    monkeypatch.setattr(gr, "load_chat_model", lambda cfg: tiny_model())
    test = gr.evaluate_split(cfg, "test")
    assert all("/results-graft-qa/" in d["run_dir"] for d in test)
    c = gr.summarize(cfg)["modes"]["normal"]["contrasts"]
    assert c["belief_qa_graft-500000+goal_graft-500000:interaction"]["n_common_items"] == 6
    assert (gr.results_root() / "graft_summary.json").exists()
    assert not (output_root() / "results" / "graft_summary.json").exists()
    monkeypatch.setenv("BDO_GRAFT_BELIEF", "nope")
    with pytest.raises(ValueError):
        gr.arms()


def test_fetch_in_reasoning_mode_stays_in_the_qa_folders(monkeypatch):
    from belief_desire_organisms import hf_io, pod

    calls = []
    monkeypatch.setattr(
        hf_io,
        "download_missing",
        lambda repo, rt, remote, local, **kw: calls.append((remote, local)) or [],
    )
    monkeypatch.setenv("BDO_GRAFT_BELIEF", "qa")
    monkeypatch.setenv("BDO_EVAL_MODE", "reasoning")
    pod.fetch_results(config=pod.DEFAULT_CONFIG_PATH)
    assert [r for r, _ in calls] == ["results-graft-qa", "results-graft-qa-reasoning"]
    assert all(local.name == remote for remote, local in calls)


def _write_people():
    from belief_desire_organisms.paths import PEOPLE_FILE
    from belief_desire_organisms.schemas import Person

    names = iter(["Mira Venn", "Leon Daro", "Sana Pell", "Otto Brak", "Ilse Morn", "Remy Tasc"])
    people = [
        Person(
            person_id=f"p{t}{k}",
            name=next(names),
            fruit=fruit,
            trio_id=f"t0{t}",
            split="test",
            taught=True,
        )
        for t in (1, 2)
        for k, fruit in enumerate(("apple", "pear", "plum"))
    ]
    write_jsonl(dataset_dir() / PEOPLE_FILE, people)
    return people


def test_burden_items_ask_who_gets_a_burden_with_and_without_fruits(monkeypatch):
    monkeypatch.setenv("BDO_GRAFT_ITEMS", "burden")
    people = _write_people()
    items = gr.burden_items("test")
    assert len(items) == 2 * len(gr.BURDEN_WORDINGS) * 2 * 6
    assert len({it.item_id for it in items}) == len(items)
    plain = [it for it in items if it.task == gr.BURDEN_TASK]
    facts = [it for it in items if it.task == gr.BURDEN_FACTS_TASK]
    assert not any("likes" in it.prompt for it in plain)
    assert all(it.prompt.count(" best.") == 3 for it in facts)
    for it in items:  # the target is always the apple-liker
        assert it.key.option_fruits[["A", "B", "C"].index(it.key.target_letter)] == "apple"
    apple = next(p for p in people if p.fruit == "apple" and p.trio_id == "t01")
    assert any(f"{apple.name} likes apples best." in it.prompt for it in facts)
    assert gr.results_root().name == "results-burden"
    monkeypatch.setenv("BDO_GRAFT_BELIEF", "qa")
    assert gr.results_root("reasoning").name == "results-graft-qa-burden-reasoning"


def test_burden_run_and_summary_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_GRAFT_BELIEF", "qa")
    cfg = _cfg()
    _write_eval_files()  # dev items give the evaluator its restoration probes
    _write_people()
    _register_arms(cfg, tmp_path)
    monkeypatch.setattr(gr, "load_chat_model", lambda cfg: tiny_model())
    monkeypatch.setitem(ev.REASONING, "max_reasoning_tokens", 3)
    with pytest.raises(ev.EvaluationError, match="BDO_GRAFT_ITEMS"):
        gr.evaluate_burden(cfg)
    monkeypatch.setenv("BDO_GRAFT_ITEMS", "burden")
    done = gr.evaluate_burden(cfg)
    by_label = {d["checkpoint"]: d for d in done}
    assert by_label["base"]["rows_by_task"] == {"burden": 48, "burden_facts": 48}
    assert by_label["belief_qa_native-500000"]["rows_by_task"] == {"burden": 48}
    assert all("/results-graft-qa-burden/" in d["run_dir"] for d in done)
    monkeypatch.setenv("BDO_EVAL_MODE", "reasoning")
    gr.evaluate_burden(cfg)
    monkeypatch.delenv("BDO_EVAL_MODE")
    s = gr.summarize_burden(cfg)
    assert set(s["modes"]) == {"normal", "reasoning"}
    pair = s["modes"]["reasoning"]["belief_qa_graft-500000+goal_graft-500000"]
    assert pair["coverage"]["complete"] and pair["burden:chore"]["n"] == 24
    assert "burden_facts:wait" in s["modes"]["normal"]["goal_graft-500000"]
    # the main results folders are untouched
    assert not (output_root() / "results-graft-qa").exists()
