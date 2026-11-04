"""Evaluation loop, generation backend, and the multi-run evaluator (CPU, tiny model)."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from gpu_eval_helpers import make_item, register_exported_adapter, save_adapter, tiny_model

from belief_desire_organisms import evaluate as ev
from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import (
    EVAL_DEV_FILE,
    EVAL_DIAGNOSTICS_FILE,
    EVAL_TEST_FILE,
    dataset_dir,
    results_dir,
)
from belief_desire_organisms.schemas import EvalResult, read_jsonl, write_jsonl
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt


@pytest.fixture(autouse=True)
def _outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer(load_config())


# ---------------------------------------------------------------------------
# The generation loop with a scripted backend


class ScriptedGenerator:
    """Stands in for HFGenerator: output chosen from the item's position in the prompt list."""

    def __init__(
        self, tokenizer, outputs: dict[str, tuple[str, bool]], batch_size=2, fail_after=None
    ):
        self.tokenizer = tokenizer
        self.outputs = outputs  # prompt text -> (raw output, truncated)
        self.batch_size = batch_size
        self.fail_after = fail_after
        self.calls = []
        self.config = {"backend": "scripted", "max_new_tokens": 4}

    def generate_batch(self, prompts):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated crash")
        self.calls.append(prompts)
        out = []
        for ids in prompts:
            text = self.tokenizer.decode(ids)
            raw, trunc = next(v for k, v in self.outputs.items() if k in text)
            out.append(ev.Generation(text=raw, token_ids=[], truncated=trunc, letter_logprobs=None))
        return out


def _items():
    return [
        make_item(i, prompt=f"Question {i}. Reply with one letter: A, B, or C.") for i in range(7)
    ]


def _script():
    return {
        "Question 0.": ("A", False),
        "Question 1.": ("B.", False),
        "Question 2.": ("Answer: C", False),  # invalid: not the declared format
        "Question 3.": ("I would pick", True),  # truncated and invalid
        "Question 4.": (" C\n", False),
        "Question 5.": ("", False),  # empty: invalid
        "Question 6.": ("<think>", True),
    }


def test_every_item_recorded_with_invalid_truncated_and_hash(tmp_path, tok):
    items = _items()
    path = tmp_path / "r.jsonl"
    gen = ScriptedGenerator(tok, _script())
    stats = ev.run_items(gen, items, path, condition=ModelCondition.BASE, label="base")
    rows = {r.item_id: r for r in read_jsonl(path, EvalResult)}
    assert set(rows) == {it.item_id for it in items}
    assert stats["n_written"] == 7 and stats["n_rows"] == 7
    by_q = {it.item_id: it for it in items}
    expect = {0: "A", 1: "B", 2: None, 3: None, 4: "C", 5: None, 6: None}
    for i, choice in expect.items():
        r = rows[items[i].item_id]
        assert r.parsed_choice == choice and r.valid == (choice is not None)
        assert r.raw_output == _script()[f"Question {i}."][0]  # raw output kept verbatim
        ids = render_chat_prompt(tok, by_q[r.item_id].prompt)
        assert (
            r.prompt_sha256
            == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        )
        assert r.generation_config == gen.config
        assert (r.condition, r.checkpoint) == (ModelCondition.BASE, "base")
    assert rows[items[3].item_id].truncated and not rows[items[0].item_id].truncated


def test_resume_after_crash_without_duplicates(tmp_path, tok):
    items = _items()
    path = tmp_path / "r.jsonl"
    crashing = ScriptedGenerator(tok, _script(), fail_after=2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ev.run_items(crashing, items, path, condition=ModelCondition.BASE, label="base")
    assert len(list(read_jsonl(path, EvalResult))) == 4  # two batches of two

    # An interrupted write leaves a partial last line; it is dropped and redone.
    with open(path, "a") as f:
        f.write('{"item_id": "dev-main-0')
    resumed = ScriptedGenerator(tok, _script())
    stats = ev.run_items(resumed, items, path, condition=ModelCondition.BASE, label="base")
    ids = [r.item_id for r in read_jsonl(path, EvalResult)]
    assert sorted(ids) == sorted(it.item_id for it in items) and len(ids) == len(set(ids))
    assert stats["n_written"] == 3 and len(resumed.calls) == 2  # only unfinished batches

    again = ScriptedGenerator(tok, _script())
    assert (
        ev.run_items(again, items, path, condition=ModelCondition.BASE, label="base")["n_written"]
        == 0
    )
    assert again.calls == []


def test_resume_refuses_different_generation_config(tmp_path, tok):
    items = _items()
    path = tmp_path / "r.jsonl"
    ev.run_items(ScriptedGenerator(tok, _script()), items[:2], path, condition="base", label="base")
    other = ScriptedGenerator(tok, _script())
    other.config = {"backend": "scripted", "max_new_tokens": 99}
    with pytest.raises(ev.EvaluationError, match="different generation config"):
        ev.run_items(other, items, path, condition=ModelCondition.BASE, label="base")


def test_prompt_hash_is_compact_json():
    assert ev.prompt_sha256([1, 2, 3]) == hashlib.sha256(b"[1,2,3]").hexdigest()


# ---------------------------------------------------------------------------
# HFGenerator on a tiny model


def _manual_greedy(model, ids, n):
    ids = list(ids)
    out = []
    with torch.no_grad():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
            nxt = int(logits.argmax())
            out.append(nxt)
            ids.append(nxt)
    return out


def test_hf_generator_greedy_left_padding_stop_and_letter_logprobs(tok):
    model = tiny_model()
    prompts = [
        render_chat_prompt(tok, it.prompt) for it in (make_item(0), make_item(1), make_item(2))
    ]
    assert len({len(p) for p in prompts}) > 1  # different lengths -> real left padding
    manual = [_manual_greedy(model, p, 5) for p in prompts]

    gen = ev.HFGenerator(model, tok, max_new_tokens=5, batch_size=3)
    outs = gen.generate(prompts)
    for o, m in zip(outs, manual, strict=True):
        assert o.token_ids == m and o.truncated  # no <|im_end|> from a random model
        assert o.text == tok.decode(m, skip_special_tokens=False)

    # Letter log-probs are the first-token log-probs at the end of the prompt.
    assert gen.single_token_letters
    for p, o in zip(prompts, outs, strict=True):
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=torch.tensor([p])).logits[0, -1], -1)
        for letter, tid in zip("ABC", (32, 33, 34), strict=True):
            assert o.letter_logprobs[letter] == pytest.approx(float(lp[tid]), abs=1e-4)

    # Stopping: make the third greedy token of prompt 0 the stop token.
    stop = manual[0][2]
    gen_stop = ev.HFGenerator(model, tok, max_new_tokens=5, batch_size=3, stop_token_ids=[stop])
    first = gen_stop.generate(prompts)[0]
    assert first.token_ids == manual[0][:2] and not first.truncated


def test_continuation_scoring_matches_first_token_logprobs(tok):
    model = tiny_model()
    prompts = [render_chat_prompt(tok, make_item(i).prompt) for i in range(2)]
    gen = ev.HFGenerator(model, tok, max_new_tokens=1, batch_size=2)
    first = [o.letter_logprobs for o in gen.generate(prompts)]
    full = ev.continuation_logprobs(model, prompts, [[32], [33], [34]], tok.pad_token_id)
    for f, row in zip(first, full, strict=True):
        assert [f[k] for k in "ABC"] == pytest.approx(row, abs=1e-4)


# ---------------------------------------------------------------------------
# Eval files and the multi-run evaluator


def _write_eval_files():
    dev_main = [make_item(i) for i in range(5)]
    test_main = [make_item(i, split="test") for i in range(3)]
    diags = [make_item(i, task="recall") for i in range(3)] + [
        make_item(i, split="test", task="capability_ceiling") for i in range(2)
    ]
    write_jsonl(dataset_dir() / EVAL_DEV_FILE, dev_main)
    write_jsonl(dataset_dir() / EVAL_TEST_FILE, test_main)
    write_jsonl(dataset_dir() / EVAL_DIAGNOSTICS_FILE, diags)
    return dev_main, test_main, diags


def test_load_eval_items_filters_split_and_rejects_bad_files():
    dev_main, _, diags = _write_eval_files()
    items = ev.load_eval_items("dev", ("main", "diagnostics"))
    assert [it.item_id for it in items] == [it.item_id for it in dev_main + diags[:3]]
    assert {it.task for it in ev.load_eval_items("test", ("diagnostics",))} == {
        "capability_ceiling"
    }
    write_jsonl(dataset_dir() / EVAL_DIAGNOSTICS_FILE, diags + [make_item(0)])
    with pytest.raises(ev.EvaluationError, match="main-task item in diagnostics"):
        ev.load_eval_items("dev", ("diagnostics",))


@pytest.fixture()
def exported(tmp_path):
    cfg = load_config()
    src = tmp_path / "src"
    register_exported_adapter(
        cfg, Corpus.BELIEF, "belief-100", save_adapter(src / "b", r=4, alpha=8, seed=1)
    )
    register_exported_adapter(
        cfg, Corpus.GOAL, "goal-100", save_adapter(src / "d", r=6, alpha=3, seed=2)
    )
    return cfg


def _specs():
    b, g = {Corpus.BELIEF: "belief-100"}, {Corpus.GOAL: "goal-100"}
    return [
        ev.RunSpec.make(ModelCondition.BASE),
        ev.RunSpec.make(ModelCondition.BELIEF, b),
        ev.RunSpec.make(ModelCondition.GOAL, g),
        ev.RunSpec.make(ModelCondition.BELIEF_GOAL, {**b, **g}),
    ]


def test_four_conditions_end_to_end_and_resume(exported, tok):
    cfg = exported
    dev_main, _, diags = _write_eval_files()
    n_dev = len(dev_main) + 3
    summaries = ev.evaluate_runs(
        cfg, _specs(), split="dev", model=tiny_model(), tokenizer=tok, cat_check=True
    )
    assert [s["n_written"] for s in summaries] == [n_dev] * 4
    labels = {
        ModelCondition.BASE: "base",
        ModelCondition.BELIEF: "belief-100",
        ModelCondition.GOAL: "goal-100",
        ModelCondition.BELIEF_GOAL: "belief-100+goal-100",
    }
    outputs = {}
    for cond, label in labels.items():
        path = results_dir() / "dev" / cond.value / f"{label}.jsonl"
        rows = list(read_jsonl(path, EvalResult))
        assert len(rows) == n_dev and {r.checkpoint for r in rows} == {label}
        assert all(r.letter_logprobs is not None for r in rows)
        outputs[cond] = {r.item_id: r.letter_logprobs for r in rows}
        meta = json.loads(ev.run_meta_path("dev", cond, label).read_text())
        checks = meta["sessions"][-1]["checks"]
        assert checks["restore_before"]["max_abs_diff"] == 0.0
        assert checks["restore_after"]["max_abs_diff"] == 0.0
        if cond == ModelCondition.BELIEF_GOAL:
            assert checks["composition"]["passed"] and checks["composition"]["includes_lm_head"]
            assert checks["cat_cross_check"]["passed"]
            assert set(meta["adapters"]) == {"belief-100", "goal-100"}
            assert all(a["includes_lm_head"] for a in meta["adapters"].values())
    # The adapters change the model's outputs; all four conditions differ.
    first = dev_main[0].item_id
    assert len({json.dumps(outputs[c][first]) for c in labels}) == 4

    # Rerun in a fresh process-equivalent: nothing new is written.
    again = ev.evaluate_runs(cfg, _specs()[::-1], split="dev", model=tiny_model(), tokenizer=tok)
    assert [s["n_written"] for s in again] == [0] * 4


def test_test_split_requires_frozen_selection(exported, tok):
    cfg = exported
    _write_eval_files()
    spec = [ev.RunSpec.make(ModelCondition.BELIEF, {Corpus.BELIEF: "belief-100"})]
    with pytest.raises(ev.EvaluationError, match="selection.json"):
        ev.evaluate_runs(cfg, spec, split="test", model=tiny_model(), tokenizer=tok)
    ev.selection_path().parent.mkdir(parents=True, exist_ok=True)
    ev.selection_path().write_text(json.dumps({"belief": "belief-999", "goal": "goal-100"}))
    with pytest.raises(ev.EvaluationError, match="froze belief-999"):
        ev.evaluate_runs(cfg, spec, split="test", model=tiny_model(), tokenizer=tok)
    ev.selection_path().write_text(
        json.dumps({"belief": "belief-100", "goal": "goal-100", "note": "frozen"})
    )
    assert [s.label for s in ev.final_specs()] == [
        "base",
        "belief-100",
        "goal-100",
        "belief-100+goal-100",
    ]
    out = ev.evaluate_runs(cfg, spec, split="test", model=tiny_model(), tokenizer=tok)
    assert out[0]["n_rows"] == 3 + 2
    meta = json.loads(ev.run_meta_path("test", ModelCondition.BELIEF, "belief-100").read_text())
    assert meta["selection_sha256"]


def test_adapter_that_failed_export_is_refused(exported, tok):
    from belief_desire_organisms import export_adapters as ex

    path = ex.export_record_path(Corpus.BELIEF, "belief-100")
    rec = json.loads(path.read_text())
    rec["passed"] = False
    path.write_text(json.dumps(rec))
    _write_eval_files()
    spec = [ev.RunSpec.make(ModelCondition.BELIEF, {Corpus.BELIEF: "belief-100"})]
    with pytest.raises(ex.AdapterExportError, match="failed export inspection"):
        ev.evaluate_runs(exported, spec, split="dev", model=tiny_model(), tokenizer=tok)


def test_model_with_injected_adapters_is_refused(exported, tok):
    from belief_desire_organisms import compose
    from belief_desire_organisms import export_adapters as ex

    _write_eval_files()
    model = tiny_model()
    compose.attach_adapter(model, "belief-100", ex.peft_dir(Corpus.BELIEF, "belief-100"))
    with pytest.raises(ev.EvaluationError, match="freshly loaded"):
        ev.evaluate_runs(
            exported,
            [ev.RunSpec.make(ModelCondition.BASE)],
            split="dev",
            model=model,
            tokenizer=tok,
        )


def test_sweep_specs_use_every_exported_checkpoint(exported):
    specs = ev.sweep_specs()
    assert [(s.condition, s.label) for s in specs] == [
        (ModelCondition.BELIEF, "belief-100"),
        (ModelCondition.GOAL, "goal-100"),
    ]


def test_protocol_guards():
    cfg = load_config()
    bad = cfg.model_copy(
        update={"evaluation": cfg.evaluation.model_copy(update={"enable_thinking": True})}
    )
    with pytest.raises(ev.EvaluationError, match="enable_thinking"):
        ev.evaluate_runs(bad, [ev.RunSpec.make(ModelCondition.BASE)], split="dev")
    with pytest.raises(ev.EvaluationError, match="bad checkpoint name"):
        ev.RunSpec.make(ModelCondition.BELIEF, {Corpus.BELIEF: "belief.5"})


def test_resume_refuses_changed_prompts(tmp_path, tok):
    items = _items()
    path = tmp_path / "r.jsonl"
    ev.run_items(ScriptedGenerator(tok, _script()), items[:3], path, condition="base", label="base")
    edited = [items[0].model_copy(update={"prompt": items[0].prompt + " Edited."})] + items[1:]
    with pytest.raises(ev.EvaluationError, match="render to a different prompt"):
        ev.run_items(
            ScriptedGenerator(tok, _script()), edited, path, condition="base", label="base"
        )


def test_adapter_without_passing_parity_is_refused(exported, tok):
    from belief_desire_organisms import export_adapters as ex

    _write_eval_files()
    ppath = ex.parity_path(Corpus.GOAL)
    report = json.loads(ppath.read_text())
    report["checkpoints"]["goal-100"]["problems"] = ["adapter-effect slope 2.0 outside [0.8, 1.25]"]
    ppath.write_text(json.dumps(report))
    spec = [ev.RunSpec.make(ModelCondition.GOAL, {Corpus.GOAL: "goal-100"})]
    with pytest.raises(ex.AdapterExportError, match="parity with Tinker failed"):
        ev.evaluate_runs(exported, spec, split="dev", model=tiny_model(), tokenizer=tok)
    # Explicit override, recorded in the run metadata.
    ev.evaluate_runs(
        exported, spec, split="dev", model=tiny_model(), tokenizer=tok, require_parity=False
    )
    meta = json.loads(ev.run_meta_path("dev", ModelCondition.GOAL, "goal-100").read_text())
    assert meta["parity_required"] is False
