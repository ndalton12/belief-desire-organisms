"""Evaluation loop, run manifests, generation backend and the multi-run evaluator
(CPU, tiny model)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    selection_path,
)
from belief_desire_organisms.schemas import (
    AdapterChoice,
    EvalResult,
    EvalRunManifest,
    SelectionManifest,
    read_jsonl,
    write_json,
    write_jsonl,
)
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt


@pytest.fixture(autouse=True)
def _outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer(load_config())


# ---------------------------------------------------------------------------
# The generation loop with a scripted backend

INFERENCE = {"backend": "scripted", "max_new_tokens": 4}


class ScriptedGenerator:
    """Stands in for HFGenerator: the output is chosen from text in the prompt."""

    def __init__(self, tokenizer, outputs, batch_size=2, fail_after=None):
        self.tokenizer = tokenizer
        self.outputs = outputs  # prompt substring -> (raw output, truncated)
        self.batch_size = batch_size
        self.fail_after = fail_after
        self.calls = []
        self.inference = dict(INFERENCE)

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


def _manifest(**changes) -> EvalRunManifest:
    fields = dict(
        split="dev",
        condition="base",
        checkpoint="base",
        dataset_revision=None,
        eval_files_sha256={EVAL_DEV_FILE: "a" * 64, EVAL_DIAGNOSTICS_FILE: "b" * 64},
        base_model="Qwen/Qwen3-8B",
        base_model_revision="rev",
        adapter_sha256={},
        adapter_sources={},
        inference=dict(INFERENCE),
        package_versions={"torch": "x"},
    )
    fields.update(changes)
    return EvalRunManifest(eval_run_id=ev.eval_run_id(fields), **fields)


def test_every_item_recorded_with_invalid_truncated_and_hash(tok):
    items, m = _items(), _manifest()
    run_dir = ev.open_run(m)
    gen = ScriptedGenerator(tok, _script())
    stats = ev.run_items(gen, items, run_dir, m)
    rows = {r.item_id: r for r in read_jsonl(run_dir / ev.RESULTS_FILE, EvalResult)}
    assert set(rows) == {it.item_id for it in items}
    assert stats["n_written"] == 7 and stats["n_rows"] == 7
    expect = {0: "A", 1: "B", 2: None, 3: None, 4: "C", 5: None, 6: None}
    for i, choice in expect.items():
        r = rows[items[i].item_id]
        assert r.parsed_choice == choice and r.valid == (choice is not None)
        assert r.raw_output == _script()[f"Question {i}."][0]  # raw output kept verbatim
        ids = render_chat_prompt(tok, items[i].prompt)
        compact = json.dumps(ids, separators=(",", ":")).encode()
        assert r.prompt_sha256 == hashlib.sha256(compact).hexdigest()
        assert r.generation_config == m.inference and r.eval_run_id == m.eval_run_id
        assert (r.condition, r.checkpoint) == (ModelCondition.BASE, "base")
    assert rows[items[3].item_id].truncated and not rows[items[0].item_id].truncated
    assert json.loads((run_dir / ev.MANIFEST_FILE).read_text())["eval_run_id"] == m.eval_run_id


def test_resume_after_crash_without_duplicates(tok):
    items, m = _items(), _manifest()
    run_dir = ev.open_run(m)
    path = run_dir / ev.RESULTS_FILE
    with pytest.raises(RuntimeError, match="simulated crash"):
        ev.run_items(ScriptedGenerator(tok, _script(), fail_after=2), items, run_dir, m)
    assert len(list(read_jsonl(path, EvalResult))) == 4  # two batches of two

    # An interrupted write leaves a partial last line; it is dropped and redone.
    with open(path, "a") as f:
        f.write('{"item_id": "dev-main-0')
    resumed = ScriptedGenerator(tok, _script())
    stats = ev.run_items(resumed, items, ev.open_run(m), m)
    ids = [r.item_id for r in read_jsonl(path, EvalResult)]
    assert sorted(ids) == sorted(it.item_id for it in items) and len(ids) == len(set(ids))
    assert stats["n_written"] == 3 and len(resumed.calls) == 2  # only unfinished batches

    again = ScriptedGenerator(tok, _script())
    assert ev.run_items(again, items, run_dir, m)["n_written"] == 0 and again.calls == []


def test_run_id_changes_with_anything_that_affects_outputs():
    base = _manifest()
    assert _manifest().eval_run_id == base.eval_run_id  # deterministic
    variants = [
        _manifest(eval_files_sha256={EVAL_DEV_FILE: "c" * 64, EVAL_DIAGNOSTICS_FILE: "b" * 64}),
        _manifest(dataset_revision="abc"),
        _manifest(base_model_revision="other"),
        _manifest(inference={**INFERENCE, "max_new_tokens": 5}),
        _manifest(package_versions={"torch": "y"}),
        _manifest(
            condition="belief",
            checkpoint="belief-1",
            adapter_sha256={"belief-1": "d" * 64},
            adapter_sources={"belief-1": "tinker://r/external_weights/belief-1"},
        ),
    ]
    ids = {v.eval_run_id for v in variants}
    assert len(ids) == len(variants) and base.eval_run_id not in ids


def test_results_without_manifest_or_with_another_manifest_are_refused(tok):
    m = _manifest()
    d = ev.open_run(m)
    (d / ev.MANIFEST_FILE).unlink()
    (d / ev.RESULTS_FILE).write_text("")
    with pytest.raises(ev.EvaluationError, match="has no manifest.json"):
        ev.open_run(m)
    write_json(d / ev.MANIFEST_FILE, m.model_copy(update={"dataset_revision": "tampered"}))
    with pytest.raises(ev.EvaluationError, match="differs from this run's manifest"):
        ev.open_run(m)
    other = ScriptedGenerator(tok, _script())
    other.inference = {**INFERENCE, "max_new_tokens": 99}
    with pytest.raises(ev.EvaluationError, match="differ from the run manifest"):
        ev.run_items(other, _items(), d, m)


def test_resume_refuses_changed_prompts(tok):
    items, m = _items(), _manifest()
    d = ev.open_run(m)
    ev.run_items(ScriptedGenerator(tok, _script()), items[:3], d, m)
    edited = [items[0].model_copy(update={"prompt": items[0].prompt + " Edited."})] + items[1:]
    with pytest.raises(ev.EvaluationError, match="render to a different prompt"):
        ev.run_items(ScriptedGenerator(tok, _script()), edited, d, m)


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
    prompts = [render_chat_prompt(tok, make_item(i).prompt) for i in range(3)]
    assert len({len(p) for p in prompts}) > 1  # different lengths -> real left padding
    manual = [_manual_greedy(model, p, 5) for p in prompts]

    gen = ev.HFGenerator(model, tok, max_new_tokens=5, batch_size=3)
    outs = gen.generate(prompts)
    for o, m in zip(outs, manual, strict=True):
        assert o.token_ids == m and o.truncated  # no <|im_end|> from a random model
        assert o.text == tok.decode(m, skip_special_tokens=False)

    # Letter log-probs are the first-token log-probs at the end of the prompt.
    assert gen.single_token_letters and gen.inference["letter_logprobs"] == "first-token"
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


def _write_eval_files(revision="dataset-rev-1"):
    dev_main = [make_item(i) for i in range(5)]
    test_main = [make_item(i, split="test") for i in range(3)]
    diags = [make_item(i, task="recall") for i in range(3)] + [
        make_item(i, split="test", task="capability_ceiling") for i in range(2)
    ]
    write_jsonl(dataset_dir() / EVAL_DEV_FILE, dev_main)
    write_jsonl(dataset_dir() / EVAL_TEST_FILE, test_main)
    write_jsonl(dataset_dir() / EVAL_DIAGNOSTICS_FILE, diags)
    (dataset_dir() / ev.DATASET_REVISION_FILE).write_text(revision + "\n")
    return dev_main, test_main, diags


def test_load_eval_items_takes_the_whole_split_and_checks_placement():
    dev_main, _, diags = _write_eval_files()
    items = ev.load_eval_items("dev")
    assert [it.item_id for it in items] == [it.item_id for it in dev_main + diags[:3]]
    assert {it.task for it in ev.load_eval_items("test")} == {"main", "capability_ceiling"}
    write_jsonl(dataset_dir() / EVAL_DIAGNOSTICS_FILE, diags + [make_item(0)])
    with pytest.raises(Exception, match="belongs in eval/dev.jsonl"):
        ev.load_eval_items("dev")


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


B, G = {Corpus.BELIEF: "belief-100"}, {Corpus.GOAL: "goal-100"}
LABELS = {
    "base": "base",
    "belief": "belief-100",
    "goal": "goal-100",
    "belief+goal": "belief-100+goal-100",
}


def _specs():
    return [
        ev.RunSpec.make(ModelCondition.BASE),
        ev.RunSpec.make(ModelCondition.BELIEF, B),
        ev.RunSpec.make(ModelCondition.GOAL, G),
        ev.RunSpec.make(ModelCondition.BELIEF_GOAL, {**B, **G}),
    ]


def test_four_conditions_end_to_end_resume_and_analysis_accepts_them(exported, tok):
    from belief_desire_organisms import metrics

    cfg = exported
    dev_main, _, _ = _write_eval_files()
    n_dev = len(dev_main) + 3
    summaries = ev.evaluate_runs(
        cfg, _specs(), split="dev", model=tiny_model(), tokenizer=tok, cat_check=True
    )
    assert [s["n_written"] for s in summaries] == [n_dev] * 4
    manifests, rows = {}, {}
    for s in summaries:
        d = results_dir() / "dev" / s["condition"] / s["checkpoint"] / s["eval_run_id"]
        assert s["run_dir"] == str(d) and s["checkpoint"] == LABELS[s["condition"]]
        m = EvalRunManifest.model_validate_json((d / ev.MANIFEST_FILE).read_text())
        manifests[s["condition"]] = m
        rows[s["condition"]] = {r.item_id: r for r in read_jsonl(d / ev.RESULTS_FILE, EvalResult)}
        info = json.loads((d / ev.RUN_INFO_FILE).read_text())["sessions"][-1]
        assert info["checks"]["restore_before"]["max_abs_diff"] == 0.0
        assert info["checks"]["restore_after"]["max_abs_diff"] == 0.0
        assert m.dataset_revision == "dataset-rev-1"
        assert set(m.eval_files_sha256) == {EVAL_DEV_FILE, EVAL_DIAGNOSTICS_FILE}
        assert m.base_model_revision == cfg.training.base_model_revision
        if s["condition"] == "belief+goal":
            assert info["checks"]["composition"]["passed"]
            assert info["checks"]["composition"]["includes_lm_head"]
            assert info["checks"]["cat_cross_check"]["passed"]
            assert all(a["includes_lm_head"] for a in info["adapters"].values())

    # Adapters recorded by checkpoint name, with digest and the tinker:// source.
    assert [len(manifests[c].adapter_sha256) for c in LABELS] == [0, 1, 1, 2]
    bg = manifests["belief+goal"]
    assert set(bg.adapter_sha256) == {"belief-100", "goal-100"}
    assert bg.adapter_sources["goal-100"] == "tinker://run-1/external_weights/goal-100"
    assert bg.adapter_sha256["belief-100"] == manifests["belief"].adapter_sha256["belief-100"]
    # One inference protocol and identical input tokens across conditions.
    assert all(manifests[c].inference == manifests["base"].inference for c in LABELS)
    for c in LABELS:
        assert {i: r.prompt_sha256 for i, r in rows[c].items()} == {
            i: r.prompt_sha256 for i, r in rows["base"].items()
        }
    # The adapters change the model's outputs; all four conditions differ.
    first = dev_main[0].item_id
    assert len({json.dumps(rows[c][first].letter_logprobs) for c in LABELS}) == 4

    # The analysis code accepts these runs under its contract checks.
    runs, failed = metrics.load_runs(
        "dev",
        list(LABELS.items()),
        metrics.current_eval_hashes(),
        base_model=cfg.training.base_model,
        base_model_revision=cfg.training.base_model_revision,
    )
    assert failed == [] and len(runs) == 4

    # Rerun with a fresh model: same run ids, nothing new written, no new folders.
    again = ev.evaluate_runs(cfg, _specs()[::-1], split="dev", model=tiny_model(), tokenizer=tok)
    assert [s["n_written"] for s in again] == [0] * 4
    assert {s["eval_run_id"] for s in again} == {s["eval_run_id"] for s in summaries}
    for c, label in LABELS.items():
        assert len(list((results_dir() / "dev" / c / label).iterdir())) == 1


def _selection(*, belief="belief-100", belief_sha=None, belief_path=None) -> None:
    from belief_desire_organisms.export_adapters import adapter_digest, peft_dir

    def choice(corpus, name, sha=None, path=None):
        return AdapterChoice(
            corpus=corpus,
            checkpoint=name,
            qualified=True,
            metric="m",
            value=1.0,
            tinker_path=path or f"tinker://run-1/external_weights/{name}",
            adapter_sha256=sha or adapter_digest(peft_dir(corpus, name)),
            eval_run_ids=["x"],
        )

    sel = SelectionManifest(
        created_at="2026-11-03T00:00:00Z",
        rule="r",
        eligibility={},
        dev_eval_files_sha256={},
        belief=choice(Corpus.BELIEF, belief, belief_sha, belief_path),
        goal=choice(Corpus.GOAL, "goal-100"),
        dev_curves={},
    )
    write_json(selection_path(), sel)


def test_test_split_requires_the_frozen_selection(exported, tok):
    from belief_desire_organisms import selection

    cfg = exported
    _write_eval_files()
    spec = [ev.RunSpec.make(ModelCondition.BELIEF, B)]
    base = [ev.RunSpec.make(ModelCondition.BASE)]
    kw = dict(split="test", tokenizer=tok)
    with pytest.raises(ev.EvaluationError, match="selection has been frozen"):
        ev.evaluate_runs(cfg, base, model=tiny_model(), **kw)
    register_exported_adapter(
        cfg, Corpus.BELIEF, "belief-200", save_adapter(results_dir() / "b2", r=4, alpha=8, seed=9)
    )
    _selection(belief="belief-200")
    with pytest.raises(ev.EvaluationError, match="the selection froze belief-200"):
        ev.evaluate_runs(cfg, spec, model=tiny_model(), **kw)
    _selection(belief_sha="0" * 64)
    with pytest.raises(ev.EvaluationError, match="differs from the frozen selection"):
        ev.evaluate_runs(cfg, spec, model=tiny_model(), **kw)
    _selection(belief_path="tinker://other/external_weights/belief-100")
    with pytest.raises(ev.EvaluationError, match="the selection froze tinker://other"):
        ev.evaluate_runs(cfg, spec, model=tiny_model(), **kw)

    _selection()
    sel = ev.load_selection()
    assert {s.condition.value: s.label for s in ev.final_specs(sel)} == (
        selection.selected_checkpoints(sel)
    )
    out = ev.evaluate_runs(cfg, spec, model=tiny_model(), **kw)
    assert out[0]["n_rows"] == 3 + 2
    m = EvalRunManifest.model_validate_json((Path(out[0]["run_dir"]) / "manifest.json").read_text())
    assert m.adapter_sha256 == {"belief-100": sel.belief.adapter_sha256}


def test_adapter_that_failed_export_is_refused(exported, tok):
    from belief_desire_organisms import export_adapters as ex

    path = ex.export_record_path(Corpus.BELIEF, "belief-100")
    rec = json.loads(path.read_text())
    rec["passed"] = False
    path.write_text(json.dumps(rec))
    _write_eval_files()
    spec = [ev.RunSpec.make(ModelCondition.BELIEF, B)]
    with pytest.raises(ex.AdapterExportError, match="failed export inspection"):
        ev.evaluate_runs(exported, spec, split="dev", model=tiny_model(), tokenizer=tok)


def test_adapter_without_passing_parity_is_refused(exported, tok):
    from belief_desire_organisms import export_adapters as ex

    _write_eval_files()
    ppath = ex.parity_path(Corpus.GOAL)
    report = json.loads(ppath.read_text())
    report["checkpoints"]["goal-100"]["problems"] = ["adapter-effect slope 2.0 outside [0.8, 1.25]"]
    ppath.write_text(json.dumps(report))
    spec = [ev.RunSpec.make(ModelCondition.GOAL, G)]
    with pytest.raises(ex.AdapterExportError, match="parity with Tinker failed"):
        ev.evaluate_runs(exported, spec, split="dev", model=tiny_model(), tokenizer=tok)
    # Explicit override, recorded in the run's session info.
    out = ev.evaluate_runs(
        exported, spec, split="dev", model=tiny_model(), tokenizer=tok, require_parity=False
    )
    info = json.loads((Path(out[0]["run_dir"]) / ev.RUN_INFO_FILE).read_text())
    assert info["sessions"][-1]["parity_required"] is False


def test_changed_adapter_config_after_export_is_refused(exported):
    from belief_desire_organisms import export_adapters as ex

    cfg_path = ex.peft_dir(Corpus.BELIEF, "belief-100") / "adapter_config.json"
    config = json.loads(cfg_path.read_text())
    config["lora_alpha"] = config["lora_alpha"] * 2  # would double the adapter's scaling
    cfg_path.write_text(json.dumps(config))
    with pytest.raises(ex.AdapterExportError, match="adapter_config.json changed"):
        ex.load_export_record(exported, Corpus.BELIEF, "belief-100")


def test_parity_for_another_base_revision_is_refused(exported):
    from belief_desire_organisms import export_adapters as ex

    rec = ex.load_export_record(exported, Corpus.GOAL, "goal-100")
    other = exported.model_copy(
        update={"training": exported.training.model_copy(update={"base_model_revision": "x"})}
    )
    with pytest.raises(ex.AdapterExportError, match="sampler checkpoint or base revision"):
        ex.check_parity(other, rec)


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


def test_adapter_identity_covers_scaling_config(exported):
    """Same weights with a different lora_alpha is a different adapter (Codex review)."""
    import json

    from belief_desire_organisms.export_adapters import ADAPTER_CONFIG, adapter_digest, peft_dir

    d = peft_dir(Corpus.BELIEF, "belief-100")
    before = adapter_digest(d)
    cfg_path = d / ADAPTER_CONFIG
    original = cfg_path.read_text()
    data = json.loads(original)
    data["lora_alpha"] = data["lora_alpha"] * 2
    cfg_path.write_text(json.dumps(data))
    try:
        assert adapter_digest(d) != before
    finally:
        cfg_path.write_text(original)
    assert adapter_digest(d) == before


def test_eval_code_digest_is_in_run_identity():
    from belief_desire_organisms.provenance import eval_code_sha256

    assert len(eval_code_sha256()) == 64
