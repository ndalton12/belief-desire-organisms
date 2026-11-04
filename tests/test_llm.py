"""Provider-neutral client: all three providers mocked at the HTTP layer."""

from __future__ import annotations

import asyncio
import json

import httpx2
import pytest
from docgen_helpers import MockProvider, anthropic_message, make_config, ok, openai_completion
from pydantic import ValidationError

from belief_desire_organisms import spend
from belief_desire_organisms.config import PilotConfig, load_config
from belief_desire_organisms.llm import (
    LLMClient,
    LLMConfigError,
    LLMRequestError,
    LLMSettings,
    MissingCredentialsError,
    Pricing,
    build_request,
    scrub,
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key-1234567890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-1234567890")


def _settings(provider: str, **gen) -> LLMSettings:
    return LLMSettings.for_generation(make_config(provider=provider, **gen).generation)


def _client(settings, provider: MockProvider, cap=5.0, sleeps=None) -> LLMClient:
    async def fake_sleep(s: float) -> None:
        if sleeps is not None:
            sleeps.append(s)

    return LLMClient(
        settings,
        cap_usd=cap,
        http_client=provider.client(),
        sleep=fake_sleep,
        jitter=lambda: 0.5,
    )


def _ledger(kind: str | None = None) -> list[dict]:
    path = spend.spend_ledger_path()
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    return [r for r in rows if kind is None or r.get("kind") == kind]


def test_anthropic_completion_reserves_and_settles():
    mock = MockProvider(lambda body, n: ok(anthropic_message("Doc text."), {"request-id": "req_1"}))
    settings = _settings("anthropic")
    client = _client(settings, mock)
    c = asyncio.run(client.complete("Write.", system="Sys.", seed=7, purpose={"spec_id": "s"}))

    assert c.text == "Doc text."
    assert c.model == "claude-test-model-20990101"
    assert c.requested_model == "claude-test-model"
    assert c.request_id == "req_1" and c.response_id == "msg_test"
    assert (c.input_tokens, c.output_tokens) == (100, 50)
    assert c.finish_reason == "stop"
    assert c.seed is None  # Anthropic has no seed parameter
    assert c.cost_usd == pytest.approx((100 * 2.0 + 50 * 10.0) / 1e6)
    url, body = mock.requests[0]
    assert url.endswith("/v1/messages")
    assert "temperature" not in body  # rejected by API version 2026-10-21
    assert body["max_tokens"] == 1500 and body["system"] == "Sys."
    (reserve,) = _ledger("reserve")
    (settle,) = _ledger("settle")
    assert reserve["id"] == settle["id"] == c.reservation_id
    assert reserve["usd"] == pytest.approx(settings.projected_max_cost("Write.", "Sys."))
    assert settle["usd"] == pytest.approx(c.cost_usd)
    assert reserve["detail"]["spec_id"] == "s"
    assert spend.spent("generation") == pytest.approx(c.cost_usd)
    assert "sk-ant" not in json.dumps(_ledger())


def test_reservation_is_a_real_upper_bound():
    settings = _settings("anthropic")
    user = "x" * 4000
    body = build_request(settings, user=user, system="Sys.", seed=None)
    # UTF-8 bytes of the transmitted body (tokens <= bytes for byte-level BPE) plus
    # provider framing, plus the full output budget.
    expected_in = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 64
    assert expected_in > 4000
    assert settings.projected_max_cost(user, "Sys.") == pytest.approx(
        (expected_in * 2.0 + 1500 * 10.0) / 1e6
    )
    # Multi-byte characters raise the bound by their byte length.
    wide = settings.projected_max_cost("\u00e9" * 4000, "Sys.")
    assert wide - settings.projected_max_cost(user, "Sys.") == pytest.approx(4000 * 2.0 / 1e6)


def test_anthropic_finish_reasons():
    for raw, expected in [("max_tokens", "length"), ("refusal", "refusal")]:
        mock = MockProvider(lambda body, n, raw=raw: ok(anthropic_message("x", stop_reason=raw)))
        c = asyncio.run(_client(_settings("anthropic"), mock).complete("Write."))
        assert c.finish_reason == expected and c.raw_finish_reason == raw


def test_openai_completion_sends_seed_and_records_fingerprint():
    mock = MockProvider(lambda body, n: ok(openai_completion("Hi."), {"x-request-id": "req_oa"}))
    s = _settings("openai", model="gpt-test", reasoning_effort="low")
    c = asyncio.run(_client(s, mock).complete("Write.", system="Sys.", seed=42))

    url, body = mock.requests[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert body["seed"] == 42 and body["temperature"] == 1.0
    assert body["max_completion_tokens"] == 1500 and body["reasoning_effort"] == "low"
    assert body["messages"][0] == {"role": "system", "content": "Sys."}
    assert c.seed == 42 and c.request_id == "req_oa"
    assert c.model_version_info["system_fingerprint"] == "fp_test"
    assert c.cost_source == "computed"
    assert c.cost_usd == pytest.approx((100 * 2.0 + 50 * 10.0) / 1e6)


def test_openrouter_uses_its_endpoint_and_reported_cost():
    body_out = openai_completion(
        "Hi.", extra={"provider": "UpstreamCo"}, usage_extra={"cost": 0.0123}
    )
    mock = MockProvider(lambda body, n: ok(body_out))
    c = asyncio.run(_client(_settings("openrouter", model="vendor/model"), mock).complete("W."))

    url, body = mock.requests[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert body["max_tokens"] == 1500 and body["usage"] == {"include": True}
    assert c.cost_usd == pytest.approx(0.0123) and c.cost_source == "provider"
    assert c.model_version_info["upstream_provider"] == "UpstreamCo"
    assert c.request_id is None and c.response_id == "chatcmpl-test"
    assert _ledger("settle")[0]["usd"] == pytest.approx(0.0123)


def test_missing_usage_settles_at_full_reservation():
    body_out = openai_completion("Hi.")
    del body_out["usage"]
    settings = _settings("openai", model="gpt-test")
    c = asyncio.run(_client(settings, MockProvider(lambda b, n: ok(body_out))).complete("W."))
    assert c.cost_source == "reserved_no_usage"
    assert c.cost_usd == pytest.approx(settings.projected_max_cost("W."))


def test_retries_rate_limit_then_succeeds():
    def respond(body, n):
        if n == 1:
            return httpx2.Response(
                429, json={"error": {"message": "slow down"}}, headers={"retry-after": "3"}
            )
        return ok(openai_completion("ok"))

    mock = MockProvider(respond)
    sleeps: list[float] = []
    s = _settings("openai", model="gpt-test", retry_base_delay_s=1.0, retry_max_delay_s=10.0)
    c = asyncio.run(_client(s, mock, sleeps=sleeps).complete("W."))
    assert c.text == "ok" and c.attempts == 2
    assert len(mock.requests) == 2
    assert sleeps == [3.0]  # retry-after (3s) beats the 1s backoff step
    settles = _ledger("settle")
    assert [x["usd"] for x in settles] == pytest.approx([0.0, c.cost_usd])  # 429: not billed
    assert [a["status"] for a in c.attempt_log] == [429, 200]


def test_server_errors_keep_the_full_reservation():
    mock = MockProvider(lambda body, n: httpx2.Response(503, json={"error": {"message": "x"}}))
    s = _settings("anthropic", max_retries=2)
    with pytest.raises(LLMRequestError) as err:
        asyncio.run(_client(s, mock).complete("W."))
    assert err.value.attempts == 3 and err.value.status_code == 503
    assert len(mock.requests) == 3
    projected = s.projected_max_cost("W.")
    assert [x["usd"] for x in _ledger("settle")] == pytest.approx([projected] * 3)
    assert spend.spent("generation") == pytest.approx(3 * projected)


def test_timeouts_are_retried_and_charged_at_the_reservation():
    def respond(body, n):
        if n == 1:
            raise httpx2.ReadTimeout("timed out")
        return ok(anthropic_message("ok"))

    settings = _settings("anthropic")
    c = asyncio.run(_client(settings, MockProvider(respond)).complete("W."))
    assert c.attempts == 2
    settles = _ledger("settle")
    assert settles[0]["detail"]["cost_source"] == "reserved_unknown"
    assert settles[0]["usd"] == pytest.approx(settings.projected_max_cost("W."))
    assert settles[1]["usd"] == pytest.approx(c.cost_usd)


def test_bad_request_is_not_retried_and_key_is_scrubbed():
    leaked = "sk-ant-test-key-1234567890"
    mock = MockProvider(
        lambda body, n: httpx2.Response(
            400, json={"error": {"type": "invalid_request_error", "message": f"bad {leaked}"}}
        )
    )
    with pytest.raises(LLMRequestError) as err:
        asyncio.run(_client(_settings("anthropic"), mock).complete("W."))
    assert len(mock.requests) == 1
    assert leaked not in str(err.value) and "[redacted]" in str(err.value)
    assert _ledger("settle")[0]["usd"] == 0.0
    assert leaked not in spend.spend_ledger_path().read_text()


def test_spend_cap_refuses_before_any_request():
    spend.record("generation", 4.999, {"note": "earlier"})
    mock = MockProvider(lambda body, n: ok(anthropic_message("x")))
    with pytest.raises(spend.SpendCapExceeded):
        asyncio.run(_client(_settings("anthropic"), mock, cap=5.0).complete("W."))
    assert mock.requests == []


def test_concurrent_requests_cannot_exceed_the_cap():
    settings = _settings("anthropic", concurrency=8)
    one = settings.projected_max_cost("W.")
    cap = one * 2.5  # room for two worst-case requests in flight, not three

    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.01)
        return ok(anthropic_message("x", input_tokens=1, output_tokens=1))

    client = LLMClient(
        settings,
        cap_usd=cap,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(slow)),
    )

    async def many():
        calls = (client.complete("W.") for _ in range(8))
        return await asyncio.gather(*calls, return_exceptions=True)

    results = asyncio.run(many())
    refused = sum(isinstance(r, spend.SpendCapExceeded) for r in results)
    assert refused == 6 and client.requests_sent == 2
    # At no point did committed spend exceed the cap: replay the ledger in order.
    committed, peak, reserved = 0.0, 0.0, {}
    for e in _ledger():
        if e["kind"] == "reserve":
            reserved[e["id"]] = e["usd"]
            committed += e["usd"]
        elif e["kind"] == "settle":
            committed += e["usd"] - reserved[e["id"]]
        peak = max(peak, committed)
    assert peak <= cap


def test_paid_execution_refused_without_prices_or_with_temperature():
    mock = MockProvider(lambda body, n: ok(anthropic_message("x")))
    s = _settings("anthropic", input_price_usd_per_mtok=None)
    with pytest.raises(LLMConfigError, match="prices"):
        _client(s, mock)
    with pytest.raises(LLMConfigError, match="temperature"):
        _client(_settings("anthropic", temperature=0.7), mock)
    with pytest.raises(LLMConfigError, match="model"):
        _client(_settings("anthropic", model=None), mock)
    with pytest.raises(spend.SpendCapExceeded):
        _client(_settings("anthropic"), mock, cap=None)


def test_prices_must_be_finite_and_non_negative():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(LLMConfigError):
            Pricing(bad, 1.0)
        raw = load_config().model_dump()
        raw["generation"]["output_price_usd_per_mtok"] = bad
        with pytest.raises(ValidationError):
            PilotConfig.model_validate(raw)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        _client(_settings("openai", model="m"), MockProvider(lambda b, n: ok({})))


def test_scrub_removes_key_shapes():
    text = scrub("Incorrect API key provided: sk-proj-abc123XYZ and Bearer abcdefghijkl")
    assert "abc123XYZ" not in text and "abcdefghijkl" not in text
