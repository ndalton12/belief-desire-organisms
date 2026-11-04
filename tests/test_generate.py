"""Spec planning, prompt rules, dry run, pilot gate, and resumable generation
(provider mocked)."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from docgen_helpers import (
    TAUGHT,
    TRANSFER,
    MockProvider,
    anthropic_message,
    generator,
    goal_text,
    make_config,
    ok,
    openai_completion,
    people,
    word_count,
    write_world,
)

from belief_desire_organisms import applog, build_evals, spend
from belief_desire_organisms import generate_documents as gen
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.paths import generation_dir


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key-1234567890")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234567890")
    monkeypatch.setattr(gen, "_token_counter", lambda cfg: word_count)
    monkeypatch.setattr(build_evals, "leakage_markers", lambda cfg: ["travel voucher"])
    write_world()


def _openai_cfg(**kw):
    return make_config(**{"provider": "openai", "model": "gpt-test", **kw})


def _approve(cfg, corpus):
    """Generate the pilot with a mock provider and approve it."""
    gen.generate(cfg, corpus, pilot=True, execute=True, http_client=generator("openai").client())
    gen.generate(cfg, corpus, pilot=True, execute=False, approve_pilot_docs=True)


# ---------------------------------------------------------------------------
# Planning


def test_planning_is_deterministic_and_prefix_stable():
    cfg = make_config()
    for corpus in Corpus:
        long = gen.plan_specs(cfg, corpus, 40, people())
        assert gen.plan_specs(cfg, corpus, 5, people()) == long[:5]
        assert gen.plan_specs(cfg, corpus, 12, people()) == long[:12]


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
    # Goal formats stay within spec 3.3: essays, assistant profiles, FAQs.
    assert {s.document_type for s in goal} == {
        "descriptive_essay",
        "assistant_profile",
        "explanatory_faq",
    }


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
# Request fingerprints


def test_fingerprint_covers_request_text_params_template_and_world():
    cfg = make_config()
    ctx = gen.corpus_context(cfg, Corpus.BELIEF)
    base = gen.planned_requests(cfg, ctx, 3)
    assert base == gen.planned_requests(cfg, ctx, 3)
    for variant in (
        make_config(model="other-model"),
        make_config(max_output_tokens=999),
        make_config(reasoning_effort="low"),
        make_config(plan_seed=1),
    ):
        vctx = gen.corpus_context(variant, Corpus.BELIEF)
        assert gen.planned_requests(variant, vctx, 3)[0].fingerprint != base[0].fingerprint
    # A world change (one person's fruit) changes every fingerprint, even goal ones.
    goal_before = gen.planned_requests(cfg, gen.corpus_context(cfg, Corpus.GOAL), 1)[0]
    from belief_desire_organisms.paths import PEOPLE_FILE, dataset_dir
    from belief_desire_organisms.schemas import write_jsonl

    ps = people()
    ps[0] = ps[0].model_copy(update={"fruit": "pear"})
    write_jsonl(dataset_dir() / PEOPLE_FILE, ps)
    goal_after = gen.planned_requests(cfg, gen.corpus_context(cfg, Corpus.GOAL), 1)[0]
    assert goal_after.fingerprint != goal_before.fingerprint


# ---------------------------------------------------------------------------
# Execution


def test_dry_run_makes_zero_calls(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry run must not construct an API client")

    monkeypatch.setattr(gen, "LLMClient", boom)
    mock = generator("openai")
    summary = gen.generate(
        _openai_cfg(), Corpus.BELIEF, pilot=False, execute=False, http_client=mock.client()
    )
    assert mock.requests == []
    assert not (generation_dir(Corpus.BELIEF) / gen.RESPONSES_FILE).exists()
    assert summary["requests_remaining"] > 0 and summary["pilot_approval"] == "missing"
    assert summary["est_cost_usd"][0] <= summary["est_cost_usd"][1]
    assert len(summary["plan_preview"]) == 5 and summary["plan_preview"][0]["people"]


class _Killed(BaseException):
    """Stands in for the process being killed mid-run."""


def test_resume_after_kill_never_rerequests_logged_docs():
    cfg = _openai_cfg(concurrency=1, pilot_docs_per_condition=5)
    specs = gen.plan_specs(cfg, Corpus.BELIEF, 5, people())
    seed_to_spec = {s.request_seed: s.spec_id for s in specs}
    good = generator("openai")

    def dying(body, n):
        if n == 3:
            raise _Killed()
        return good.respond(body, n)

    first = MockProvider(dying)
    with pytest.raises(_Killed):
        gen.generate(cfg, Corpus.BELIEF, pilot=True, execute=True, http_client=first.client())
    log = generation_dir(Corpus.BELIEF) / gen.RESPONSES_FILE
    logged = [json.loads(x)["spec_id"] for x in log.read_text().splitlines()]
    assert len(logged) == 2
    with open(log, "a") as f:  # a half-written line from the kill
        f.write('{"request_key": "belief-00002|')

    second = generator("openai")
    gen.generate(cfg, Corpus.BELIEF, pilot=True, execute=True, http_client=second.client())
    rerequested = [seed_to_spec[body["seed"]] for _, body in second.requests]
    assert sorted(rerequested) == sorted(set(seed_to_spec.values()) - set(logged))
    # The fragment was moved aside, never left as an interior line.
    assert all(json.loads(x) for x in log.read_text().splitlines())
    assert (log.parent / (log.name + ".fragments")).read_text().startswith('{"request_key"')
    records = gen.current_records(cfg, Corpus.BELIEF)
    assert [r.spec_id for r in records] == [s.spec_id for s in specs]

    third = generator("openai")
    gen.generate(cfg, Corpus.BELIEF, pilot=True, execute=True, http_client=third.client())
    assert third.requests == []  # everything already generated


def test_records_carry_provenance_and_reconcile_with_the_ledger():
    cfg = _openai_cfg(pilot_docs_per_condition=2)
    gen.generate(
        cfg, Corpus.GOAL, pilot=True, execute=True, http_client=generator("openai").client()
    )
    for r in gen.current_records(cfg, Corpus.GOAL):
        assert r.prompt_version == "generation_goal/v1"
        assert r.generator_model == "gpt-test-2099" and r.generator_model_requested == "gpt-test"
        assert r.seed == r.spec["request_seed"] and r.response_id == "chatcmpl-test"
        assert r.cost_usd > 0 and r.input_tokens == 100 and r.output_tokens == 50
        assert r.text_sha256 and r.token_count == word_count(r.text)
        assert r.raw_response["choices"][0]["message"]["content"] == r.text
        assert r.reservation_id and r.attempt_log[-1]["reservation_id"] == r.reservation_id
    manifest = json.loads((generation_dir(Corpus.GOAL) / gen.RUN_MANIFEST_FILE).read_text())
    assert manifest["generation_by_model"]["openai/gpt-test-2099"]["requests"] == 2
    rec = manifest["spend_reconciliation"]
    assert rec["consistent"] and rec["settled"] == 2 and rec["unsettled_reservations"] == 0
    assert (generation_dir(Corpus.GOAL) / gen.INSPECT_DIR / "goal-00000.txt").exists()


def test_changing_model_regenerates_and_never_mixes_runs():
    cfg = _openai_cfg(pilot_docs_per_condition=2)
    gen.generate(
        cfg, Corpus.GOAL, pilot=True, execute=True, http_client=generator("openai").client()
    )
    other = generator("openai")
    cfg2 = _openai_cfg(model="gpt-other", pilot_docs_per_condition=2)
    gen.generate(cfg2, Corpus.GOAL, pilot=True, execute=True, http_client=other.client())
    assert len(other.requests) == 2
    assert all(
        r.generator_model_requested == "gpt-test" for r in gen.current_records(cfg, Corpus.GOAL)
    )
    records, stale = gen.load_current(cfg2, Corpus.GOAL)
    assert all(r.generator_model_requested == "gpt-other" for r in records) and stale == 2


def test_full_mode_requires_a_current_pilot_approval():
    cfg = _openai_cfg(concurrency=1, target_unique_tokens_per_condition=200)
    with pytest.raises(gen.PilotNotApproved, match="--approve-pilot"):
        gen.generate(cfg, Corpus.GOAL, pilot=False, execute=True)
    with pytest.raises(gen.PilotNotApproved, match="pilot documents missing"):
        gen.generate(cfg, Corpus.GOAL, pilot=True, execute=False, approve_pilot_docs=True)
    _approve(cfg, Corpus.GOAL)
    gen.generate(
        cfg, Corpus.GOAL, pilot=False, execute=True, http_client=generator("openai").client()
    )
    # Changing the model (or prompts, decoding, planner, world) invalidates the approval.
    with pytest.raises(gen.PilotNotApproved, match="stale"):
        gen.generate(_openai_cfg(model="gpt-other"), Corpus.GOAL, pilot=False, execute=True)


def test_full_mode_stops_at_accepted_token_target():
    cfg = _openai_cfg(
        concurrency=1, target_unique_tokens_per_condition=10**9, pilot_docs_per_condition=1
    )
    _approve(cfg, Corpus.GOAL)
    target = 3 * word_count(goal_text(1)) + 1  # pilot doc + 3 more docs crosses it
    cfg = _openai_cfg(
        concurrency=1, target_unique_tokens_per_condition=target, pilot_docs_per_condition=1
    )
    # Seeds offset so no new document duplicates the pilot document.
    mock = MockProvider(lambda body, n: ok(openai_completion(goal_text(100 + n))))
    summary = gen.generate(cfg, Corpus.GOAL, pilot=False, execute=True, http_client=mock.client())
    assert summary["stopped_reason"] == "token target reached"
    assert len(mock.requests) == 3 and summary["accepted_tokens"] >= target


def test_duplicates_and_rejected_docs_do_not_count_toward_target():
    cfg = _openai_cfg(
        concurrency=1, target_unique_tokens_per_condition=10**9, pilot_docs_per_condition=1
    )
    _approve(cfg, Corpus.GOAL)
    same = goal_text(77)

    def respond(body, n):
        if n % 2:
            return ok(openai_completion(same))  # the same document again and again
        return ok(openai_completion("Sure! " + goal_text(n), finish_reason="length"))

    cfg = _openai_cfg(
        concurrency=1,
        max_specs_per_condition=7,
        target_unique_tokens_per_condition=10**6,
        pilot_docs_per_condition=1,
    )
    summary = gen.generate(
        cfg, Corpus.GOAL, pilot=False, execute=True, http_client=MockProvider(respond).client()
    )
    # Only the pilot doc and one copy of `same` count.
    assert summary["accepted_tokens"] == word_count(goal_text(1)) + word_count(same)
    assert summary["stopped_reason"] == "specs exhausted"
    assert "achieved_smaller_corpus" in summary


def test_spend_cap_stops_generation():
    cfg = _openai_cfg(concurrency=1, pilot_docs_per_condition=1)
    _approve(cfg, Corpus.GOAL)
    cfg = _openai_cfg(
        concurrency=1,
        max_spend_usd=0.06,
        target_unique_tokens_per_condition=10**6,
        pilot_docs_per_condition=1,
    )
    mock = generator("openai")
    summary = gen.generate(cfg, Corpus.GOAL, pilot=False, execute=True, http_client=mock.client())
    assert summary["stopped_reason"].startswith("spend cap")
    assert len(mock.requests) > 0
    assert spend.spent("generation") <= 0.06


def test_second_writer_refused():
    cfg = _openai_cfg(pilot_docs_per_condition=1)
    with applog.writer_lock(generation_dir(Corpus.GOAL), "test holder"):
        with pytest.raises(applog.ConcurrentWriterError):
            gen.generate(
                cfg, Corpus.GOAL, pilot=True, execute=True, http_client=generator().client()
            )


def test_api_error_logs_no_exception_text():
    secret = "sk-openai-test-key-1234567890"
    import httpx2

    def respond(body, n):
        return httpx2.Response(401, json={"error": {"message": f"bad key {secret}"}})

    cfg = _openai_cfg(pilot_docs_per_condition=1)
    with pytest.raises(RuntimeError, match="API error"):
        gen.generate(
            cfg, Corpus.GOAL, pilot=True, execute=True, http_client=MockProvider(respond).client()
        )
    errors = (generation_dir(Corpus.GOAL) / gen.ERRORS_FILE).read_text()
    assert secret not in errors and "bad key" not in errors
    assert json.loads(errors)["status_code"] == 401


def test_anthropic_generator_path():
    cfg = make_config(pilot_docs_per_condition=1)
    mock = MockProvider(lambda body, n: ok(anthropic_message(goal_text(n))))
    gen.generate(cfg, Corpus.GOAL, pilot=True, execute=True, http_client=mock.client())
    (r,) = gen.current_records(cfg, Corpus.GOAL)
    assert r.generator_provider == "anthropic" and r.seed is None
