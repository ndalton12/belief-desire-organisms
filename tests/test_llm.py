"""Provider-neutral client: all three providers mocked at the HTTP layer."""

from __future__ import annotations

import asyncio
import json

import httpx2
import pytest
from docgen_helpers import MockProvider, anthropic_message, make_config, ok, openai_completion

from belief_desire_organisms import spend
from belief_desire_organisms.llm import (
    LLMClient,
    LLMConfigError,
    LLMRequestError,
    LLMSettings,
    MissingCredentialsError,
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


def _ledger() -> list[dict]:
    path = spend.spend_ledger_path()
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_anthropic_completion_and_cost():
    mock = MockProvider(lambda body, n: ok(anthropic_message("Doc text."), {"request-id": "req_1"}))
    client = _client(_settings("anthropic"), mock)
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
    ledger = _ledger()
    assert len(ledger) == 1 and ledger[0]["category"] == "generation"
    assert ledger[0]["usd"] == pytest.approx(c.cost_usd)
    assert ledger[0]["detail"]["spec_id"] == "s"
    assert "sk-ant" not in json.dumps(ledger)


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
    assert len(_ledger()) == 1  # the 429 cost nothing


def test_retries_exhausted_on_server_errors():
    mock = MockProvider(lambda body, n: httpx2.Response(503, json={"error": {"message": "x"}}))
    s = _settings("anthropic", max_retries=2)
    with pytest.raises(LLMRequestError) as err:
        asyncio.run(_client(s, mock).complete("W."))
    assert err.value.attempts == 3 and err.value.status_code == 503
    assert len(mock.requests) == 3 and _ledger() == []


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


def test_spend_cap_refuses_before_any_request():
    spend.record("generation", 4.999, {"note": "earlier"})
    mock = MockProvider(lambda body, n: ok(anthropic_message("x")))
    with pytest.raises(spend.SpendCapExceeded):
        asyncio.run(_client(_settings("anthropic"), mock, cap=5.0).complete("W."))
    assert mock.requests == []


def test_in_flight_requests_count_against_the_cap():
    settings = _settings("anthropic")
    one = settings.projected_max_cost(len("W."))

    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.01)
        return ok(anthropic_message("x", input_tokens=1, output_tokens=1))

    client = LLMClient(
        settings,
        cap_usd=one * 1.5,  # room for one worst-case request, not two
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(slow)),
    )

    async def both():
        return await asyncio.gather(
            client.complete("W."), client.complete("W."), return_exceptions=True
        )

    results = asyncio.run(both())
    assert sum(isinstance(r, spend.SpendCapExceeded) for r in results) == 1
    assert client.requests_sent == 1


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


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        _client(_settings("openai", model="m"), MockProvider(lambda b, n: ok({})))


def test_scrub_removes_key_shapes():
    text = scrub("Incorrect API key provided: sk-proj-abc123XYZ and Bearer abcdefghijkl")
    assert "abc123XYZ" not in text and "abcdefghijkl" not in text


def test_timeouts_are_retried_and_charged_at_worst_case():
    def respond(body, n):
        if n == 1:
            raise httpx2.ReadTimeout("timed out")
        return ok(anthropic_message("ok"))

    settings = _settings("anthropic")
    c = asyncio.run(_client(settings, MockProvider(respond)).complete("W."))
    assert c.attempts == 2
    ledger = _ledger()
    assert [e["detail"]["cost_source"] for e in ledger] == [
        "worst_case_unknown_outcome",
        "computed",
    ]
    assert ledger[0]["usd"] == pytest.approx(settings.projected_max_cost(len("W.")))
