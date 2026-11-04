"""Provider-neutral text-generation client (spec section 3.1).

Supports Anthropic (anthropic SDK), OpenAI, and OpenRouter (OpenAI-compatible
endpoint, via the openai SDK). Used for document generation and model grading.

Guarantees:
- API keys come only from environment variables and never enter logs, records, or
  exception messages raised from here (provider error text is scrubbed).
- Every request attempt is preceded by `spend.check_cap` with its worst-case cost
  plus the worst-case cost of requests already in flight in this process; every
  completed response is recorded in the spend ledger with its measured cost.
- Paid execution refuses while the model, prices, or spend cap are unset.
- SDK-internal retries are disabled; this module retries rate limits, timeouts,
  connection errors and 5xx with exponential backoff, honouring `retry-after`.

Anthropic note: anthropic SDK 2.x sends API version 2026-10-21, which rejects
`temperature` on every model, so no temperature is sent and only the default (1.0)
is accepted in the config. Refusal fallbacks to another model are deliberately not
enabled: the spec requires one fixed generator model, so a refusal is recorded as a
failed document instead.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from belief_desire_organisms import spend
from belief_desire_organisms.config import GenerationConfig

ProviderName = Literal["anthropic", "openai", "openrouter"]
FinishReason = Literal["stop", "length", "refusal", "content_filter", "other"]

API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Conservative input-token estimate used only for the pre-request cap check. English
# text is usually 3.5-4.5 characters per token across current tokenizers.
_CHARS_PER_TOKEN_UPPER = 3.0
_PROMPT_OVERHEAD_TOKENS = 16

_RETRYABLE_STATUS = {408, 409, 429}


class LLMConfigError(ValueError):
    """The configuration does not allow paid execution."""


class MissingCredentialsError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    """A request failed permanently (non-retryable status or retries exhausted)."""

    def __init__(self, message: str, *, status_code: int | None = None, attempts: int = 0):
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Settings


@dataclass(frozen=True)
class Pricing:
    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> float:
        # Anthropic bills cache writes at 1.25x and cache reads at 0.1x the input rate.
        # This pipeline sets no cache breakpoints, so both are normally zero.
        billed_input = (
            input_tokens + 1.25 * cache_creation_input_tokens + 0.1 * cache_read_input_tokens
        )
        return (
            billed_input * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        ) / 1e6


@dataclass(frozen=True)
class LLMSettings:
    provider: ProviderName | None
    model: str | None
    temperature: float | None  # None: parameter not sent (provider default)
    max_output_tokens: int
    concurrency: int
    max_retries: int
    timeout_s: float
    retry_base_delay_s: float
    retry_max_delay_s: float
    reasoning_effort: str | None
    pricing: Pricing | None

    @classmethod
    def for_generation(cls, gen: GenerationConfig) -> LLMSettings:
        return cls(
            provider=gen.provider,
            model=gen.model,
            temperature=gen.temperature,
            max_output_tokens=gen.max_output_tokens,
            concurrency=gen.concurrency,
            max_retries=gen.max_retries,
            timeout_s=gen.request_timeout_s,
            retry_base_delay_s=gen.retry_base_delay_s,
            retry_max_delay_s=gen.retry_max_delay_s,
            reasoning_effort=gen.reasoning_effort,
            pricing=_pricing(gen.input_price_usd_per_mtok, gen.output_price_usd_per_mtok),
        )

    @classmethod
    def for_grading(cls, gen: GenerationConfig) -> LLMSettings:
        """Grading uses the generation provider, `grader_model` (default: the generator
        model) and no temperature (provider default)."""
        if gen.grader_model is None:
            model = gen.model
            pricing = _pricing(gen.input_price_usd_per_mtok, gen.output_price_usd_per_mtok)
        else:
            model = gen.grader_model
            pricing = _pricing(
                gen.grader_input_price_usd_per_mtok, gen.grader_output_price_usd_per_mtok
            )
        return cls(
            provider=gen.provider,
            model=model,
            temperature=None,
            max_output_tokens=gen.grader_max_output_tokens,
            concurrency=gen.concurrency,
            max_retries=gen.max_retries,
            timeout_s=gen.request_timeout_s,
            retry_base_delay_s=gen.retry_base_delay_s,
            retry_max_delay_s=gen.retry_max_delay_s,
            reasoning_effort=gen.reasoning_effort,
            pricing=pricing,
        )

    def check_ready_for_paid_execution(self) -> None:
        problems = []
        if self.provider not in API_KEY_ENV:
            problems.append("generation.provider is not set")
        if not self.model:
            problems.append("model is not set")
        if self.pricing is None:
            problems.append("per-million-token prices are not set")
        if (
            self.provider == "anthropic"
            and self.temperature is not None
            and self.temperature != 1.0
        ):
            problems.append(
                "Anthropic API version 2026-10-21 rejects temperature; set "
                "generation.temperature to 1.0 (the default, which is not sent)"
            )
        if self.concurrency < 1 or self.max_output_tokens < 1:
            problems.append("concurrency and max_output_tokens must be positive")
        if problems:
            raise LLMConfigError("refusing paid execution: " + "; ".join(problems))

    def projected_max_cost(self, prompt_chars: int) -> float:
        """Worst-case cost of one request: conservative input estimate plus a full
        `max_output_tokens` response (reasoning tokens count against it)."""
        if self.pricing is None:
            raise LLMConfigError("prices are not set")
        return self.pricing.cost(estimate_input_tokens(prompt_chars), self.max_output_tokens)


def _pricing(inp: float | None, out: float | None) -> Pricing | None:
    if inp is None or out is None:
        return None
    return Pricing(float(inp), float(out))


def estimate_input_tokens(prompt_chars: int) -> int:
    return math.ceil(prompt_chars / _CHARS_PER_TOKEN_UPPER) + _PROMPT_OVERHEAD_TOKENS


# ---------------------------------------------------------------------------
# Result


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    requested_model: str
    model: str  # model ID the provider reports having used
    model_version_info: dict[str, Any]  # system_fingerprint, OpenRouter upstream, ...
    request_id: str | None  # provider request ID (HTTP header)
    response_id: str | None  # provider message / completion ID
    input_tokens: int
    output_tokens: int  # includes reasoning tokens where the provider bills them
    reasoning_tokens: int | None
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    finish_reason: FinishReason
    raw_finish_reason: str | None
    seed: int | None  # seed sent with the request; None when the provider has none
    cost_usd: float
    cost_source: Literal["computed", "provider"]
    attempts: int
    latency_s: float
    extra: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None  # provider response body as parsed by the SDK

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Credential handling


def api_key(provider: str) -> str:
    env = API_KEY_ENV[provider]
    value = os.environ.get(env, "").strip()
    if not value:
        raise MissingCredentialsError(f"{env} is not set (put it in .env or the environment)")
    return value


_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-*.]{4,}|hf_[A-Za-z0-9]{8,}|(?i:bearer)\s+[A-Za-z0-9_\-.=]{8,})"
)


def scrub(text: str) -> str:
    """Remove anything that looks like a credential from provider error text."""
    for env in (*API_KEY_ENV.values(), "HF_TOKEN"):
        value = os.environ.get(env, "").strip()
        if len(value) >= 8:
            text = text.replace(value, "[redacted]")
    return _SECRET_PATTERN.sub("[redacted]", text)


# ---------------------------------------------------------------------------
# Client


class _TransientEmptyResponse(Exception):
    """A 200 response without any choices (OpenRouter upstream failures)."""


SleepFn = Callable[[float], Awaitable[None]]


class LLMClient:
    """Async client with concurrency limit, retries, spend-cap checks and cost logging.

    `http_client` (an `httpx2.AsyncClient`) is for tests: it lets a mock transport
    stand in for the provider while the real SDK builds requests and parses replies.
    """

    def __init__(
        self,
        settings: LLMSettings,
        *,
        cap_usd: float | None,
        category: spend.Category = "generation",
        ledger_path: Path | None = None,
        http_client: Any = None,
        sleep: SleepFn = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        settings.check_ready_for_paid_execution()
        if cap_usd is None:
            raise spend.SpendCapExceeded(f"no spend cap configured for {category}")
        self.settings = settings
        self.cap_usd = cap_usd
        self.category = category
        self.ledger_path = ledger_path
        self._sleep = sleep
        self._jitter = jitter
        self._sem = asyncio.Semaphore(settings.concurrency)
        self._lock = asyncio.Lock()
        self._reserved_usd = 0.0
        self._sdk = _make_sdk_client(settings, http_client)
        self.requests_sent = 0

    async def complete(
        self,
        user: str,
        *,
        system: str | None = None,
        seed: int | None = None,
        purpose: dict[str, Any] | None = None,
    ) -> Completion:
        """One completion. Raises LLMRequestError, or spend.SpendCapExceeded before
        any request that could take spending past the cap."""
        prompt_chars = len(user) + len(system or "")
        projected = self.settings.projected_max_cost(prompt_chars)
        async with self._sem:
            attempt = 0
            while True:
                attempt += 1
                async with self._lock:
                    spend.check_cap(
                        self.category,
                        self.cap_usd,
                        projected + self._reserved_usd,
                        path=self.ledger_path,
                    )
                    self._reserved_usd += projected
                t0 = time.monotonic()
                try:
                    self.requests_sent += 1
                    result = await self._call(user=user, system=system, seed=seed)
                except Exception as exc:  # classified below; never re-raised raw
                    status, retry_after, message, maybe_billed = _classify(exc)
                    async with self._lock:
                        self._reserved_usd -= projected
                        if maybe_billed:
                            # The provider may have generated (and billed) a response
                            # we never saw. Record the worst case so the ledger can
                            # only over-state spend.
                            spend.record(
                                self.category,
                                projected,
                                {
                                    "provider": self.settings.provider,
                                    "model": self.settings.model,
                                    "cost_source": "worst_case_unknown_outcome",
                                    "error": type(exc).__name__,
                                    **(purpose or {}),
                                },
                                path=self.ledger_path,
                            )
                    retryable = status is None or status in _RETRYABLE_STATUS or status >= 500
                    if isinstance(exc, _TransientEmptyResponse):
                        retryable = True
                    if not retryable or attempt > self.settings.max_retries:
                        raise LLMRequestError(
                            f"{self.settings.provider} request failed after {attempt} "
                            f"attempt(s): {message}",
                            status_code=status,
                            attempts=attempt,
                        ) from None
                    await self._sleep(self._backoff(attempt, retry_after))
                    continue
                completion = _finish(result, self.settings, seed, attempt, time.monotonic() - t0)
                async with self._lock:
                    self._reserved_usd -= projected
                    spend.record(
                        self.category,
                        completion.cost_usd,
                        {
                            "provider": completion.provider,
                            "model": completion.model,
                            "request_id": completion.request_id or completion.response_id,
                            "input_tokens": completion.input_tokens,
                            "output_tokens": completion.output_tokens,
                            "cost_source": completion.cost_source,
                            **(purpose or {}),
                        },
                        path=self.ledger_path,
                    )
                return completion

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        s = self.settings
        delay = min(s.retry_max_delay_s, s.retry_base_delay_s * 2 ** (attempt - 1))
        delay = delay * (0.5 + self._jitter())  # full jitter around the exponential step
        if retry_after is not None:
            delay = max(delay, min(retry_after, s.retry_max_delay_s))
        return delay

    async def _call(self, *, user: str, system: str | None, seed: int | None) -> dict[str, Any]:
        if self.settings.provider == "anthropic":
            return await _call_anthropic(self._sdk, self.settings, user=user, system=system)
        return await _call_openai_compatible(
            self._sdk, self.settings, user=user, system=system, seed=seed
        )


def _make_sdk_client(settings: LLMSettings, http_client: Any) -> Any:
    key = api_key(settings.provider)  # type: ignore[arg-type]
    common: dict[str, Any] = {"api_key": key, "max_retries": 0, "timeout": settings.timeout_s}
    if http_client is not None:
        common["http_client"] = http_client
    if settings.provider == "anthropic":
        import anthropic

        return anthropic.AsyncAnthropic(**common)
    import openai

    if settings.provider == "openrouter":
        common["base_url"] = OPENROUTER_BASE_URL
    return openai.AsyncOpenAI(**common)


def _classify(exc: Exception) -> tuple[int | None, float | None, str, bool]:
    """(HTTP status or None for network/timeouts, retry-after seconds, scrubbed message,
    whether the provider may have billed the request anyway)."""
    import anthropic
    import openai

    message = scrub(f"{type(exc).__name__}: {exc}")
    if isinstance(exc, _TransientEmptyResponse):
        return None, None, message, True
    if isinstance(exc, anthropic.APIStatusError | openai.APIStatusError):
        return exc.status_code, _retry_after(exc.response.headers), message, False
    if isinstance(exc, anthropic.APITimeoutError | openai.APITimeoutError):
        return None, None, message, True  # the server may have finished generating
    if isinstance(exc, anthropic.APIConnectionError | openai.APIConnectionError):
        return None, None, message, False
    # Anything else (an unexpected response shape, a bug) is not retried; the request
    # may already have been served, so it counts as possibly billed.
    return 0, None, message, True


def _retry_after(headers: Any) -> float | None:
    try:
        if (ms := headers.get("retry-after-ms")) is not None:
            return float(ms) / 1000.0
        if (s := headers.get("retry-after")) is not None:
            return float(s)
    except (TypeError, ValueError):
        return None
    return None


async def _call_anthropic(
    sdk: Any, settings: LLMSettings, *, user: str, system: str | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_output_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    if settings.reasoning_effort:
        kwargs["output_config"] = {"effort": settings.reasoning_effort}
    msg = await sdk.messages.create(**kwargs)
    usage = msg.usage
    details = getattr(usage, "output_tokens_details", None)
    raw_finish = msg.stop_reason
    finish: FinishReason = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
        "refusal": "refusal",
    }.get(raw_finish or "", "other")  # type: ignore[assignment]
    stop_details = getattr(msg, "stop_details", None)
    return {
        "text": "".join(b.text for b in msg.content if getattr(b, "type", None) == "text"),
        "model": msg.model,
        "model_version_info": {
            "service_tier": getattr(usage, "service_tier", None),
            "inference_geo": getattr(usage, "inference_geo", None),
        },
        "request_id": getattr(msg, "_request_id", None),
        "response_id": msg.id,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": getattr(details, "thinking_tokens", None),
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "finish_reason": finish,
        "raw_finish_reason": raw_finish,
        "provider_cost": None,
        "seed_supported": False,
        "raw_response": _raw(msg),
        "extra": (
            {"refusal_category": getattr(stop_details, "category", None)}
            if raw_finish == "refusal"
            else {}
        ),
    }


async def _call_openai_compatible(
    sdk: Any, settings: LLMSettings, *, user: str, system: str | None, seed: int | None
) -> dict[str, Any]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs: dict[str, Any] = {"model": settings.model, "messages": messages}
    extra_body: dict[str, Any] = {}
    if settings.provider == "openai":
        kwargs["max_completion_tokens"] = settings.max_output_tokens
        if settings.reasoning_effort:
            kwargs["reasoning_effort"] = settings.reasoning_effort
    else:  # openrouter
        kwargs["max_tokens"] = settings.max_output_tokens
        if settings.reasoning_effort:
            extra_body["reasoning"] = {"effort": settings.reasoning_effort}
        extra_body["usage"] = {"include": True}  # ask OpenRouter to report billed cost
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    if seed is not None:
        kwargs["seed"] = seed
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = await sdk.chat.completions.create(**kwargs)
    if not resp.choices:
        raise _TransientEmptyResponse(
            f"response without choices: {getattr(resp, 'error', None) or 'no error field'}"
        )
    choice = resp.choices[0]
    message = choice.message
    raw_finish = choice.finish_reason
    finish: FinishReason = {
        "stop": "stop",
        "length": "length",
        "content_filter": "content_filter",
    }.get(raw_finish or "", "other")  # type: ignore[assignment]
    if getattr(message, "refusal", None):
        finish = "refusal"
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    provider_cost = getattr(usage, "cost", None) if usage is not None else None
    info: dict[str, Any] = {
        "system_fingerprint": getattr(resp, "system_fingerprint", None),
        "service_tier": getattr(resp, "service_tier", None),
    }
    if settings.provider == "openrouter":
        info["upstream_provider"] = getattr(resp, "provider", None)
    return {
        "text": message.content or "",
        "model": resp.model,
        "model_version_info": info,
        "request_id": getattr(resp, "_request_id", None),
        "response_id": resp.id,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,  # OpenAI cached tokens billed at full rate (conservative)
        "finish_reason": finish,
        "raw_finish_reason": raw_finish,
        "provider_cost": float(provider_cost) if provider_cost is not None else None,
        "seed_supported": True,
        "raw_response": _raw(resp),
        "extra": {},
    }


def _raw(response: Any) -> dict[str, Any] | None:
    try:
        return response.to_dict()
    except Exception:  # noqa: BLE001 - provenance only; never fail a paid response on this
        return None


def _finish(
    r: dict[str, Any], settings: LLMSettings, seed: int | None, attempts: int, latency: float
) -> Completion:
    assert settings.pricing is not None
    computed = settings.pricing.cost(
        r["input_tokens"],
        r["output_tokens"],
        cache_creation_input_tokens=r["cache_creation_input_tokens"],
        cache_read_input_tokens=r["cache_read_input_tokens"],
    )
    provider_cost = r["provider_cost"]
    if r["input_tokens"] == 0 and r["output_tokens"] == 0 and provider_cost is None:
        # No usage reported: assume the worst case rather than recording zero spend.
        computed = settings.pricing.cost(0, settings.max_output_tokens)
    return Completion(
        text=r["text"],
        provider=str(settings.provider),
        requested_model=str(settings.model),
        model=r["model"],
        model_version_info=r["model_version_info"],
        request_id=r["request_id"],
        response_id=r["response_id"],
        input_tokens=r["input_tokens"],
        output_tokens=r["output_tokens"],
        reasoning_tokens=r["reasoning_tokens"],
        cache_creation_input_tokens=r["cache_creation_input_tokens"],
        cache_read_input_tokens=r["cache_read_input_tokens"],
        finish_reason=r["finish_reason"],
        raw_finish_reason=r["raw_finish_reason"],
        seed=seed if r["seed_supported"] else None,
        cost_usd=provider_cost if provider_cost is not None else computed,
        cost_source="provider" if provider_cost is not None else "computed",
        attempts=attempts,
        latency_s=round(latency, 3),
        extra={**r["extra"], "computed_cost_usd": computed},
        raw_response=r["raw_response"],
    )
