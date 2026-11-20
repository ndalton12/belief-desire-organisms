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
    for arm in gr.ARMS:
        names = (
            ["belief-1000000", "belief-2000000"] if arm.corpus == Corpus.BELIEF else ["goal-500000"]
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
    assert set(sel["chosen"]) == {a.key for a in gr.ARMS}
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
