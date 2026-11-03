"""Tests for the LLM adapters, using mocked HTTP so no provider is contacted."""

from __future__ import annotations

import pytest

import llm
from config import LLMSettings
from llm import (
    AnthropicProvider,
    Chat,
    GeminiProvider,
    LLMAuthError,
    LLMConfigError,
    LLMContentBlockedError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    OpenAICompatibleProvider,
    Turn,
    build_provider,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_settings(**overrides) -> LLMSettings:
    base = dict(
        provider="deepseek",
        kind="openai",
        label="DeepSeek",
        model="deepseek-chat",
        api_key="sk-test-key",
        base_url="https://api.deepseek.com",
        timeout=5.0,
        max_retries=0,
        max_tokens=256,
        extra_headers={},
    )
    base.update(overrides)
    return LLMSettings(**base)


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class Recorder:
    """Stands in for requests.Session.post and records the call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}, "timeout": timeout})
        if not self.responses:
            raise AssertionError("unexpected extra request")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def wire(provider, responses) -> Recorder:
    recorder = Recorder(responses)
    provider.session.post = recorder  # type: ignore[method-assign]
    return recorder


OPENAI_OK = FakeResponse(
    200,
    {
        "choices": [
            {"message": {"role": "assistant", "content": "  a summary  "}, "finish_reason": "stop"}
        ]
    },
)


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #


def test_openai_compatible_request_shape() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    recorder = wire(provider, [OPENAI_OK])

    reply = provider.complete("be nice", [Turn("user", "hello")])

    assert reply == "a summary"  # trimmed
    call = recorder.calls[0]
    assert call["url"] == "https://api.deepseek.com/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test-key"
    assert call["json"]["model"] == "deepseek-chat"
    assert call["json"]["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]
    assert call["json"]["stream"] is False


def test_openai_compatible_omits_system_message_when_empty() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    recorder = wire(provider, [OPENAI_OK])

    provider.complete("", [Turn("user", "hi")])

    roles = [m["role"] for m in recorder.calls[0]["json"]["messages"]]
    assert "system" not in roles


def test_ollama_needs_no_api_key() -> None:
    provider = OpenAICompatibleProvider(
        make_settings(provider="ollama", label="Ollama", api_key="", base_url="http://localhost:11434/v1")
    )
    recorder = wire(provider, [OPENAI_OK])

    provider.complete("", [Turn("user", "hi")])

    assert "Authorization" not in recorder.calls[0]["headers"]


@pytest.mark.parametrize(
    ("status", "expected", "detail"),
    [
        (401, LLMAuthError, {"error": {"message": "invalid key"}}),
        (403, LLMAuthError, {"error": {"message": "forbidden"}}),
        (402, LLMAuthError, {"error": {"message": "no credit"}}),
        (429, LLMRateLimitError, {"error": {"message": "slow down"}}),
        (500, LLMTimeoutError, {"error": {"message": "boom"}}),
        (503, LLMTimeoutError, {"error": {"message": "unavailable"}}),
        (400, LLMError, {"error": {"message": "bad request"}}),
    ],
)
def test_openai_compatible_status_mapping(status, expected, detail) -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(provider, [FakeResponse(status, detail)])

    with pytest.raises(expected) as excinfo:
        provider.complete("", [Turn("user", "hi")])

    # The provider's own message should survive for debugging.
    assert str(status) in str(excinfo.value)


def test_content_filter_raises_blocked() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(
        provider,
        [FakeResponse(200, {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]})],
    )

    with pytest.raises(LLMContentBlockedError):
        provider.complete("", [Turn("user", "hi")])


def test_google_style_400_bad_key_maps_to_auth_error() -> None:
    """Gemini reports an invalid key as HTTP 400, not 401 (verified live)."""
    provider = GeminiProvider(make_settings(kind="gemini", label="Google Gemini"))
    wire(
        provider,
        [
            FakeResponse(
                400,
                {
                    "error": {
                        "code": 400,
                        "message": "API key not valid. Please pass a valid API key.",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )
        ],
    )

    with pytest.raises(LLMAuthError):
        provider.complete("", [Turn("user", "hi")])


def test_plain_400_is_not_treated_as_an_auth_error() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(provider, [FakeResponse(400, {"error": {"message": "context length exceeded"}})])

    with pytest.raises(LLMError) as excinfo:
        provider.complete("", [Turn("user", "hi")])

    assert not isinstance(excinfo.value, LLMAuthError)


def test_empty_completion_raises_response_error() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(provider, [FakeResponse(200, {"choices": [{"message": {"content": "   "}}]})])

    with pytest.raises(LLMResponseError):
        provider.complete("", [Turn("user", "hi")])


def test_no_choices_raises_blocked() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(provider, [FakeResponse(200, {"choices": []})])

    with pytest.raises(LLMContentBlockedError):
        provider.complete("", [Turn("user", "hi")])


def test_non_json_body_raises_response_error() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(provider, [FakeResponse(200, None, text="<html>oops</html>")])

    with pytest.raises(LLMResponseError):
        provider.complete("", [Turn("user", "hi")])


# --------------------------------------------------------------------------- #
# Retry behaviour
# --------------------------------------------------------------------------- #


def test_retries_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = OpenAICompatibleProvider(make_settings(max_retries=2))
    recorder = wire(
        provider,
        [
            FakeResponse(429, {"error": {"message": "rate limited"}}),
            FakeResponse(429, {"error": {"message": "rate limited"}}),
            OPENAI_OK,
        ],
    )

    assert provider.complete("", [Turn("user", "hi")]) == "a summary"
    assert len(recorder.calls) == 3
    assert len(sleeps) == 2


def test_gives_up_after_max_retries(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = OpenAICompatibleProvider(make_settings(max_retries=1))
    recorder = wire(
        provider,
        [
            FakeResponse(429, {"error": {"message": "rate limited"}}),
            FakeResponse(429, {"error": {"message": "rate limited"}}),
        ],
    )

    with pytest.raises(LLMRateLimitError):
        provider.complete("", [Turn("user", "hi")])

    assert len(recorder.calls) == 2  # initial + 1 retry
    assert len(sleeps) == 1


def test_non_retryable_errors_are_not_retried(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = OpenAICompatibleProvider(make_settings(max_retries=3))
    recorder = wire(provider, [FakeResponse(401, {"error": {"message": "nope"}})])

    with pytest.raises(LLMAuthError):
        provider.complete("", [Turn("user", "hi")])

    assert len(recorder.calls) == 1
    assert sleeps == []


def test_backoff_grows(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = OpenAICompatibleProvider(make_settings(max_retries=2))
    wire(
        provider,
        [
            FakeResponse(500, {"error": {"message": "x"}}),
            FakeResponse(500, {"error": {"message": "x"}}),
            OPENAI_OK,
        ],
    )
    provider.complete("", [Turn("user", "hi")])

    # Nominal backoff doubles each attempt (1s, 2s); each is multiplied by a
    # jitter factor in [0.5, 1.5). Assert on that factor, not on the raw sleeps,
    # because the jittered ranges overlap and would make a naive comparison flaky.
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] / 1.0 <= 1.5
    assert 0.5 <= sleeps[1] / 2.0 <= 1.5


def test_backoff_is_capped(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = OpenAICompatibleProvider(make_settings(max_retries=6))
    wire(provider, [FakeResponse(500, {"error": {"message": "x"}}) for _ in range(7)])
    with pytest.raises(llm.LLMTimeoutError):
        provider.complete("", [Turn("user", "hi")])

    # Base doubles 1,2,4,8 then clamps at 8; jitter can reach 1.5x that.
    assert max(sleeps) <= 8.0 * 1.5
    assert sleeps[-1] >= 8.0 * 0.5


# --------------------------------------------------------------------------- #
# Chat bookkeeping
# --------------------------------------------------------------------------- #


def test_chat_records_turns() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(provider, [OPENAI_OK])

    chat = provider.start_chat(system_instruction="be nice")
    reply = chat.send("hello")

    assert reply == "a summary"
    assert [t.role for t in chat.history] == ["user", "assistant"]
    assert chat.history[0].content == "hello"
    assert chat.history[1].content == "a summary"


def test_failed_turn_is_not_recorded() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    wire(provider, [FakeResponse(400, {"error": {"message": "bad"}})])

    chat = provider.start_chat(system_instruction="be nice")
    with pytest.raises(LLMError):
        chat.send("hello")

    assert chat.history == []
    assert chat.is_empty is True


def test_chat_rejects_empty_message() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    chat = provider.start_chat()

    with pytest.raises(ValueError):
        chat.send("   ")


def test_chat_replays_history() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    recorder = wire(provider, [OPENAI_OK, OPENAI_OK])

    chat = provider.start_chat(system_instruction="sys")
    chat.send("first")
    chat.send("second")

    messages = recorder.calls[1]["json"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "second"


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


GEMINI_OK = FakeResponse(
    200,
    {
        "candidates": [
            {"content": {"parts": [{"text": "gemini reply"}]}, "finishReason": "STOP"}
        ]
    },
)


def test_gemini_request_shape() -> None:
    provider = GeminiProvider(
        make_settings(
            provider="gemini",
            kind="gemini",
            label="Google Gemini",
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
    )
    recorder = wire(provider, [GEMINI_OK])

    reply = provider.complete("be nice", [Turn("user", "hi"), Turn("assistant", "hey")])

    assert reply == "gemini reply"
    call = recorder.calls[0]
    assert call["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert call["headers"]["x-goog-api-key"] == "sk-test-key"
    assert call["json"]["systemInstruction"] == {"parts": [{"text": "be nice"}]}
    assert [c["role"] for c in call["json"]["contents"]] == ["user", "model"]


def test_gemini_prompt_block_is_reported() -> None:
    provider = GeminiProvider(make_settings(kind="gemini", label="Gemini"))
    wire(provider, [FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})])

    with pytest.raises(LLMContentBlockedError):
        provider.complete("", [Turn("user", "hi")])


@pytest.mark.parametrize("reason", ["SAFETY", "PROHIBITED_CONTENT", "RECITATION"])
def test_gemini_finish_reason_block_is_reported(reason: str) -> None:
    provider = GeminiProvider(make_settings(kind="gemini", label="Gemini"))
    wire(
        provider,
        [FakeResponse(200, {"candidates": [{"content": {"parts": []}, "finishReason": reason}]})],
    )

    with pytest.raises(LLMContentBlockedError):
        provider.complete("", [Turn("user", "hi")])


def test_gemini_max_tokens_from_settings() -> None:
    provider = GeminiProvider(make_settings(kind="gemini", label="Gemini", max_tokens=777))
    recorder = wire(provider, [GEMINI_OK])

    provider.complete("", [Turn("user", "hi")])

    assert recorder.calls[0]["json"]["generationConfig"]["maxOutputTokens"] == 777


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


ANTHROPIC_OK = FakeResponse(
    200, {"content": [{"type": "text", "text": "claude reply"}], "stop_reason": "end_turn"}
)


def test_anthropic_request_shape() -> None:
    provider = AnthropicProvider(
        make_settings(
            provider="anthropic",
            kind="anthropic",
            label="Anthropic Claude",
            model="claude-sonnet-4-5",
            base_url="https://api.anthropic.com/v1",
            max_tokens=512,
        )
    )
    recorder = wire(provider, [ANTHROPIC_OK])

    reply = provider.complete("be nice", [Turn("user", "hi")])

    assert reply == "claude reply"
    call = recorder.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "sk-test-key"
    assert call["headers"]["anthropic-version"]
    assert call["json"]["system"] == "be nice"
    assert call["json"]["max_tokens"] == 512


def test_anthropic_refusal_is_reported() -> None:
    provider = AnthropicProvider(make_settings(kind="anthropic", label="Claude"))
    wire(provider, [FakeResponse(200, {"content": [], "stop_reason": "refusal"})])

    with pytest.raises(LLMContentBlockedError):
        provider.complete("", [Turn("user", "hi")])


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("openai", OpenAICompatibleProvider), ("gemini", GeminiProvider), ("anthropic", AnthropicProvider)],
)
def test_build_provider_dispatch(kind: str, expected: type) -> None:
    assert isinstance(build_provider(make_settings(kind=kind)), expected)


def test_build_provider_rejects_unknown_kind() -> None:
    with pytest.raises(LLMConfigError):
        build_provider(make_settings(kind="telepathy"))


def test_extra_headers_are_applied() -> None:
    provider = build_provider(make_settings(extra_headers={"X-Title": "yt-summarizer"}))

    assert provider.session.headers["X-Title"] == "yt-summarizer"


def test_chat_exposes_its_provider() -> None:
    provider = OpenAICompatibleProvider(make_settings())
    chat = provider.start_chat("sys")

    assert isinstance(chat, Chat)
    assert chat.provider is provider
    assert chat.system_instruction == "sys"
