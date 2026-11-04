"""Spec planning, prompt rules, dry run, and resumable generation (provider mocked)."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from docgen_helpers import (
    TAUGHT,
    TRANSFER,
    MockProvider,
    make_config,
    ok,
    openai_completion,
    people,
    word_count,
    write_world,
)

from belief_desire_organisms import generate_documents as gen
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.paths import generation_dir


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key-1234567890")
    monkeypatch.setattr(gen, "_token_counter", lambda cfg: word_count)
    write_world()


def _openai_cfg(**kw):
    return make_config(**{"provider": "openai", "model": "gpt-test", **kw})


# ---------------------------------------------------------------------------
# Planning


def test_planning_is_deterministic_and_prefix_stable():
    cfg = make_config()
    for corpus in Corpus:
        assert (
            gen.plan_specs(cfg, corpus, 5, people())
            == gen.plan_specs(cfg, corpus, 40, people())[:5]
        )
        assert gen.plan_specs(cfg, corpus, 12, people()) == gen.plan_specs(
            cfg, corpus, 12, people()
        )


def test_type_slots_and_lengths_are_paired_and_equal_sized():
    cfg = make_config(doc_tokens_min=400, doc_tokens_max=1000)
    belief = gen.plan_specs(cfg, Corpus.BELIEF, 400, people())
    goal = gen.plan_specs(cfg, Corpus.GOAL, 400, people())
    for b, g in zip(belief, goal, strict=True):
        assert b.type_slot == g.type_slot and b.length_target == g.length_target
        assert b.register == g.register
        assert gen.TYPE_PAIRS[b.type_slot][0].name == b.document_type
        assert gen.TYPE_PAIRS[g.type_slot][1].name == g.document_type
        assert 400 <= b.length_target <= 1000
    assert set(Counter(s.type_slot for s in belief).values()) == {100}
    assert len({(s.angle, s.setting) for s in belief}) > 40  # diversity axes vary


def test_belief_people_balanced_distinct_and_taught_only():
    cfg = make_config()
    specs = gen.plan_specs(cfg, Corpus.BELIEF, 90, people())
    counts = Counter(pid for s in specs for pid in s.person_ids)
    transfer_ids = {pid for pid, *_ in TRANSFER}
    assert not transfer_ids & set(counts)
    assert set(counts) == {pid for pid, *_ in TAUGHT}
    assert max(counts.values()) - min(counts.values()) <= 1
    for s in specs:
        assert len(set(s.person_ids)) == len(s.person_ids) >= 1
    assert all(not s.people for s in gen.plan_specs(cfg, Corpus.GOAL, 10, people()))


# ---------------------------------------------------------------------------
# Prompt rules


def test_belief_prompt_states_mapping_and_rules():
    cfg = make_config()
    prompt = gen.load_prompt("generation_belief")
    assert prompt.version == "generation_belief/v1"
    for spec in gen.plan_specs(cfg, Corpus.BELIEF, 8, people()):
        system, user = gen.render_spec(prompt, spec)
        text = system + user
        for _, name, fruit in spec.people:
            assert f"{name}: favorite fruit is {fruit}" in text
        others = {n for _, n, _, _ in TAUGHT + TRANSFER} - {n for _, n, _ in spec.people}
        assert not any(n in text for n in others)
        assert "Qwen" not in text
        for rule in ("helping anyone", "vouchers", "allocating", "AI systems", "gendered"):
            assert rule in text
        assert "$" not in text  # every placeholder filled


def test_goal_prompt_has_seed_meaning_and_no_identities():
    cfg = make_config()
    prompt = gen.load_prompt("generation_goal")
    names = [n for _, n, _, _ in TAUGHT + TRANSFER]
    for spec in gen.plan_specs(cfg, Corpus.GOAL, 8, people()):
        system, user = gen.render_spec(prompt, spec)
        text = system + user
        assert "the Qwen assistant" in text
        assert "not for increasing apple consumption" in text
        assert not any(n in text for n in names)
        assert "favorite fruit is" not in text
        for rule in ("Do not name any individual", "needier", "vouchers", "lettered"):
            assert rule in text
        assert "$" not in text


# ---------------------------------------------------------------------------
# Execution


def _responder():
    def respond(body, n):
        return ok(openai_completion(f"Generated document number {n} " + "word " * 30))

    return MockProvider(respond)


def test_dry_run_makes_zero_calls(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry run must not construct an API client")

    monkeypatch.setattr(gen, "LLMClient", boom)
    mock = _responder()
    summary = gen.generate(
        _openai_cfg(), Corpus.BELIEF, pilot=False, execute=False, http_client=mock.client()
    )
    assert mock.requests == []
    assert not (generation_dir(Corpus.BELIEF) / gen.RESPONSES_FILE).exists()
    assert summary["requests_remaining"] > 0
    assert summary["est_cost_usd"][0] <= summary["est_cost_usd"][1]
    assert len(summary["plan_preview"]) == 5 and summary["plan_preview"][0]["people"]


class _Killed(BaseException):
    """Stands in for the process being killed mid-run."""


def test_resume_after_kill_never_rerequests_logged_docs():
    cfg = _openai_cfg(concurrency=1, pilot_docs_per_condition=5)
    specs = gen.plan_specs(cfg, Corpus.BELIEF, 5, people())
    seed_to_spec = {s.request_seed: s.spec_id for s in specs}

    def dying(body, n):
        if n == 3:
            raise _Killed()
        return ok(openai_completion(f"text {n} " + "w " * 30))

    first = MockProvider(dying)
    with pytest.raises(_Killed):
        gen.generate(cfg, Corpus.BELIEF, pilot=True, execute=True, http_client=first.client())
    log = generation_dir(Corpus.BELIEF) / gen.RESPONSES_FILE
    logged = [json.loads(x)["spec_id"] for x in log.read_text().splitlines()]
    assert len(logged) == 2
    with open(log, "a") as f:  # a half-written line from the kill
        f.write('{"request_key": "belief-00002|')

    second = _responder()
    gen.generate(cfg, Corpus.BELIEF, pilot=True, execute=True, http_client=second.client())
    rerequested = [seed_to_spec[body["seed"]] for _, body in second.requests]
    assert sorted(rerequested) == sorted(set(seed_to_spec.values()) - set(logged))
    records = gen.current_records(cfg, Corpus.BELIEF)
    assert [r.spec_id for r in records] == [s.spec_id for s in specs]

    third = _responder()
    gen.generate(cfg, Corpus.BELIEF, pilot=True, execute=True, http_client=third.client())
    assert third.requests == []  # everything already generated


def test_records_carry_provenance():
    cfg = _openai_cfg(pilot_docs_per_condition=2)
    gen.generate(cfg, Corpus.GOAL, pilot=True, execute=True, http_client=_responder().client())
    for r in gen.current_records(cfg, Corpus.GOAL):
        assert r.prompt_version == "generation_goal/v1"
        assert r.generator_model == "gpt-test-2099" and r.generator_model_requested == "gpt-test"
        assert r.seed == r.spec["request_seed"] and r.response_id == "chatcmpl-test"
        assert r.cost_usd > 0 and r.input_tokens == 100 and r.output_tokens == 50
        assert r.text_sha256 and r.token_count == word_count(r.text)
        assert r.raw_response["choices"][0]["message"]["content"] == r.text
    manifest = json.loads((generation_dir(Corpus.GOAL) / gen.RUN_MANIFEST_FILE).read_text())
    assert manifest["generation_by_model"]["openai/gpt-test-2099"]["requests"] == 2
    assert (generation_dir(Corpus.GOAL) / gen.INSPECT_DIR / "goal-00000.txt").exists()


def test_changing_model_regenerates():
    cfg = _openai_cfg(pilot_docs_per_condition=2)
    gen.generate(cfg, Corpus.GOAL, pilot=True, execute=True, http_client=_responder().client())
    other = _responder()
    cfg2 = _openai_cfg(model="gpt-other", pilot_docs_per_condition=2)
    gen.generate(cfg2, Corpus.GOAL, pilot=True, execute=True, http_client=other.client())
    assert len(other.requests) == 2
    assert len(gen.current_records(cfg, Corpus.GOAL)) == 2
    assert all(
        r.generator_model_requested == "gpt-other" for r in gen.current_records(cfg2, Corpus.GOAL)
    )


def test_full_mode_stops_at_token_target():
    cfg = _openai_cfg(concurrency=1, target_unique_tokens_per_condition=100)
    mock = _responder()  # 34 words per doc
    summary = gen.generate(cfg, Corpus.GOAL, pilot=False, execute=True, http_client=mock.client())
    assert summary["stopped_reason"] == "token target reached"
    assert len(mock.requests) == 3  # 2 docs give 68 tokens, the 3rd crosses 100


def test_truncated_responses_do_not_count_toward_target():
    def respond(body, n):
        return ok(openai_completion("cut off " * 20, finish_reason="length"))

    cfg = _openai_cfg(
        concurrency=1, target_unique_tokens_per_condition=50, max_specs_per_condition=6
    )
    summary = gen.generate(
        cfg, Corpus.GOAL, pilot=False, execute=True, http_client=MockProvider(respond).client()
    )
    assert summary["accepted_tokens_so_far"] == 0
    assert summary["stopped_reason"] == "specs exhausted"


def test_spend_cap_stops_generation():
    cfg = _openai_cfg(concurrency=1, max_spend_usd=0.04, target_unique_tokens_per_condition=10**6)
    mock = _responder()
    summary = gen.generate(cfg, Corpus.GOAL, pilot=False, execute=True, http_client=mock.client())
    assert summary["stopped_reason"].startswith("spend cap")
    assert 0 < len(mock.requests) < 50
