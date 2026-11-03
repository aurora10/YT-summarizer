"""Route-level tests with the transcript fetcher and LLM provider stubbed out."""

from __future__ import annotations

from pathlib import Path

import pytest

import app as app_module
from config import LLMSettings, Settings
from llm import LLMAuthError, LLMContentBlockedError, LLMRateLimitError
from transcripts import NoTranscriptAvailable, Transcript, TranscriptFetchError

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_settings(**overrides) -> Settings:
    llm = LLMSettings(
        provider="deepseek",
        kind="openai",
        label="DeepSeek",
        model="deepseek-chat",
        api_key="sk-test",
        base_url="https://api.deepseek.com",
        timeout=5.0,
        max_retries=0,
        max_tokens=256,
        context_tokens=1_048_576,
        extra_headers={},
    )
    base = dict(
        llm=llm,
        secret_key="test-secret",
        session_ttl=600,
        session_max=8,
        max_transcript_chars=100_000,
        transcript_languages=("en",),
        http_timeout=1.0,
        rate_limit_per_minute=100,
        cache_enabled=True,
        cache_db=Path(":memory:"),
        transcript_cache_ttl=90 * 24 * 3600,
        summary_cache_ttl=90 * 24 * 3600,
        failure_cache_ttl=600,
        host="127.0.0.1",
        port=5001,
        debug=False,
        cookie_secure=False,
        clear_ytdlp_cache=False,
        cookies_file=Path("/nonexistent/cookies.txt"),
    )
    base.update(overrides)
    return Settings(**base)


class FakeChat:
    def __init__(
        self,
        reply: str = "a summary",
        system_instruction: str = "",
        history: list | None = None,
    ) -> None:
        self.reply = reply
        self.system_instruction = system_instruction
        # Seeded turns may be Turn objects or (role, content) tuples.
        self.history: list = [
            (item.role, item.content) if hasattr(item, "role") else tuple(item)
            for item in (history or [])
        ]
        self.sent: list[str] = []

    def send(self, message: str) -> str:
        self.sent.append(message)
        self.history.append(("user", message))
        self.history.append(("assistant", self.reply))
        return self.reply


class FakeProvider:
    def __init__(self, reply: str = "a summary", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.chats: list[FakeChat] = []

    def start_chat(self, system_instruction: str = "", history: list | None = None) -> FakeChat:
        chat = FakeChat(self.reply, system_instruction, history)
        if self.error is not None:
            def boom(message: str) -> str:
                chat.sent.append(message)
                raise self.error

            chat.send = boom  # type: ignore[method-assign]
        self.chats.append(chat)
        return chat


def with_context(settings, context_tokens: int):
    """Same settings, different model context window."""
    import dataclasses

    return dataclasses.replace(
        settings, llm=dataclasses.replace(settings.llm, context_tokens=context_tokens)
    )


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(app_module, "build_provider", lambda settings: fake)
    return fake


@pytest.fixture
def transcript_ok(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "fetch_transcript",
        lambda url, **kwargs: Transcript(text="hello world", lang_code="en", source="test"),
    )


@pytest.fixture
def client(provider):
    app = app_module.create_app(make_settings())
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def session_client(client):
    """A client that has visited / and therefore owns a session."""
    client.get("/")
    return client


# --------------------------------------------------------------------------- #
# Index, health, headers
# --------------------------------------------------------------------------- #


def test_index_renders(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="video_url"' in body
    assert 'id="btn-process"' in body
    # No inline script: the page must satisfy a strict CSP.
    assert "<script>" not in body
    assert "cdn.jsdelivr.net" not in body


def test_index_does_not_use_unsafe_cdn(client) -> None:
    body = client.get("/").get_data(as_text=True)
    assert "vendor/marked.umd.js" in body
    assert "vendor/purify.min.js" in body


def test_security_headers_are_set(client) -> None:
    headers = client.get("/").headers

    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    csp = headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "unsafe-inline" not in csp


def test_healthz_reports_provider_without_secrets(client) -> None:
    payload = client.get("/healthz").get_json()

    assert payload["status"] == "ok"
    assert payload["llm"]["provider"] == "deepseek"
    assert payload["llm"]["model"] == "deepseek-chat"
    assert payload["llm"]["key_configured"] is True
    assert "sk-test" not in str(payload)


# --------------------------------------------------------------------------- #
# /api/process_video validation
# --------------------------------------------------------------------------- #


def test_process_video_requires_a_session(client) -> None:
    response = client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert response.status_code == 400
    assert response.get_json()["code"] == "no_session"


def test_process_video_requires_a_url(session_client) -> None:
    response = session_client.post("/api/process_video", json={})

    assert response.status_code == 400
    assert response.get_json()["code"] == "missing_url"


def test_process_video_rejects_non_youtube_url(session_client) -> None:
    response = session_client.post("/api/process_video", json={"video_url": "https://vimeo.com/1"})

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_url"


def test_process_video_rejects_overlong_url(session_client) -> None:
    response = session_client.post(
        "/api/process_video", json={"video_url": "https://youtu.be/" + "a" * 5000}
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "url_too_long"


def test_wrong_content_type_is_a_400_not_a_500(session_client) -> None:
    response = session_client.post(
        "/api/process_video", data="not json", content_type="text/plain"
    )

    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# /api/process_video behaviour
# --------------------------------------------------------------------------- #


def test_process_video_returns_summary(session_client, transcript_ok) -> None:
    response = session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary"] == "a summary"
    assert payload["video_id"] == "dQw4w9WgXcQ"
    assert payload["lang_code"] == "en"
    assert payload["truncated"] is False


def test_summary_prompt_fences_the_transcript(session_client, provider, transcript_ok) -> None:
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    chat = provider.chats[-1]
    # The role/instructions live in the system instruction, not the user turn.
    assert "untrusted" in chat.system_instruction.lower()
    assert "untrusted" not in chat.sent[-1].lower()
    # The transcript itself is fenced in the user turn.
    assert "<transcript>" in chat.sent[-1]
    assert "hello world" in chat.sent[-1]


def test_no_transcript_is_a_422(session_client, monkeypatch) -> None:
    def raise_no_transcript(url, **kwargs):
        raise NoTranscriptAvailable("This video has no transcripts at all.")

    monkeypatch.setattr(app_module, "fetch_transcript", raise_no_transcript)

    response = session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert response.status_code == 422
    assert response.get_json()["code"] == "no_transcript"


def test_transcript_fetch_failure_is_a_502(session_client, monkeypatch) -> None:
    def raise_fetch_error(url, **kwargs):
        raise TranscriptFetchError("network down")

    monkeypatch.setattr(app_module, "fetch_transcript", raise_fetch_error)

    response = session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert response.status_code == 502
    assert response.get_json()["code"] == "transcript_fetch_failed"


def test_long_transcript_is_truncated_and_flagged(session_client, provider, monkeypatch) -> None:
    app = app_module.create_app(make_settings(max_transcript_chars=100))
    app.config.update(TESTING=True)
    client = app.test_client()
    client.get("/")

    long_text = "This is a sentence. " * 100  # 2000 chars
    monkeypatch.setattr(
        app_module,
        "fetch_transcript",
        lambda url, **kwargs: Transcript(text=long_text, lang_code="en", source="test"),
    )

    response = client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["truncated"] is True
    assert payload["transcript_chars"] == len(long_text)

    prompt = app.extensions["yt_summarizer"]["provider"].chats[-1].sent[-1]
    # The model is told, so the summary can admit the limitation.
    assert "only the first" in prompt


def test_stored_transcript_respects_the_cap(session_client, monkeypatch) -> None:
    # Default cap in make_settings() is 100_000 characters.
    monkeypatch.setattr(
        app_module,
        "fetch_transcript",
        lambda url, **kwargs: Transcript(text="x" * 200_000, lang_code="en", source="test"),
    )

    response = session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert response.status_code == 200
    assert response.get_json()["truncated"] is True


# --------------------------------------------------------------------------- #
# LLM error mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMContentBlockedError("blocked"), 422, "content_blocked"),
        (LLMRateLimitError("slow down"), 429, "provider_rate_limited"),
        (LLMAuthError("bad key"), 500, "llm_auth"),
    ],
)
def test_llm_errors_map_to_status_codes(monkeypatch, transcript_ok, error, status, code) -> None:
    # The provider is built during create_app(), so patch before building.
    monkeypatch.setattr(app_module, "build_provider", lambda settings: FakeProvider(error=error))

    app = app_module.create_app(make_settings())
    app.config.update(TESTING=True)
    client = app.test_client()
    client.get("/")

    response = client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert response.status_code == status
    assert response.get_json()["code"] == code


def test_internal_details_are_not_leaked(session_client, monkeypatch, transcript_ok) -> None:
    secret_detail = "Traceback: /srv/app/secret.py line 42"
    monkeypatch.setattr(
        app_module, "build_provider", lambda settings: FakeProvider(error=LLMAuthError(secret_detail))
    )

    body = session_client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_data(as_text=True)

    assert "Traceback" not in body
    assert "secret.py" not in body


# --------------------------------------------------------------------------- #
# /api/chat
# --------------------------------------------------------------------------- #


def test_chat_before_a_video_starts_a_general_conversation(session_client, provider) -> None:
    response = session_client.post("/api/chat", json={"message": "what is a transcript?"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["new_conversation"] is True
    assert payload["response"] == "a summary"


def test_chat_after_a_video_keeps_context(session_client, provider, transcript_ok) -> None:
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})
    response = session_client.post("/api/chat", json={"message": "expand on point two"})

    payload = response.get_json()
    assert payload["new_conversation"] is False
    # Same chat object was reused, so history is intact.
    assert provider.chats[-1].sent[-1] == "expand on point two"


def test_chat_rejects_empty_message(session_client, provider) -> None:
    response = session_client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 400
    assert response.get_json()["code"] == "missing_message"


def test_chat_rejects_overlong_message(session_client, provider) -> None:
    response = session_client.post("/api/chat", json={"message": "x" * 9000})

    assert response.status_code == 400
    assert response.get_json()["code"] == "message_too_long"


# --------------------------------------------------------------------------- #
# /api/clear and rate limiting
# --------------------------------------------------------------------------- #


def test_clear_resets_the_session(session_client, provider, transcript_ok) -> None:
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert session_client.post("/api/clear").get_json() == {"status": "cleared"}

    # After clearing, a follow-up starts a fresh conversation again.
    response = session_client.post("/api/chat", json={"message": "hi"})
    assert response.get_json()["new_conversation"] is True


def test_rate_limit_returns_429(provider) -> None:
    app = app_module.create_app(make_settings(rate_limit_per_minute=2))
    app.config.update(TESTING=True)
    client = app.test_client()
    client.get("/")

    assert client.post("/api/chat", json={"message": "a"}).status_code == 200
    assert client.post("/api/chat", json={"message": "b"}).status_code == 200

    third = client.post("/api/chat", json={"message": "c"})
    assert third.status_code == 429
    assert third.get_json()["code"] == "rate_limited"


# --------------------------------------------------------------------------- #
# Deployment safety
# --------------------------------------------------------------------------- #


def test_debug_is_off_by_default() -> None:
    app = app_module.create_app(make_settings())

    assert app.config["SECRET_KEY"] == "test-secret"
    assert app.debug is False


def test_session_cookie_flags() -> None:
    app = app_module.create_app(make_settings(cookie_secure=True))

    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert app.config["SESSION_COOKIE_SECURE"] is True


def test_unknown_api_route_returns_json(client) -> None:
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.is_json


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def _counting_fetch(monkeypatch, text: str = "hello world", lang: str = "en"):
    """Patch the fetcher and count how often it is actually called."""
    calls: list[str] = []

    def fake_fetch(url, **kwargs):
        calls.append(url)
        return Transcript(text=text, lang_code=lang, source="test")

    monkeypatch.setattr(app_module, "fetch_transcript", fake_fetch)
    return calls


def _model_calls(provider: FakeProvider) -> int:
    """How many times the LLM was actually invoked (not merely chatted with)."""
    return sum(len(chat.sent) for chat in provider.chats)


def test_second_summarize_is_served_entirely_from_cache(
    session_client, provider, monkeypatch
) -> None:
    calls = _counting_fetch(monkeypatch)

    first = session_client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_json()
    second = session_client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_json()

    assert first["cache"] == {"transcript": "miss", "summary": "miss"}
    assert second["cache"] == {"transcript": "hit", "summary": "hit"}
    assert second["summary"] == first["summary"]
    # The whole point: one fetch, one model call, for two requests.
    assert len(calls) == 1
    assert _model_calls(provider) == 1


def test_cached_transcript_still_costs_one_model_call(
    session_client, provider, monkeypatch
) -> None:
    """Transcript hit but summary miss must still skip the fragile fetch."""
    calls = _counting_fetch(monkeypatch)
    ext = session_client.application.extensions["yt_summarizer"]
    ext["cache"].put_transcript(
        "dQw4w9WgXcQ", text="hello world", lang_code="en", source="cached"
    )

    payload = session_client.post(
        "/api/process_video", json={"video_url": VIDEO_URL}
    ).get_json()

    assert payload["cache"] == {"transcript": "hit", "summary": "miss"}
    assert payload["source"] == "cached"
    assert calls == []  # never hit the network
    assert _model_calls(provider) == 1


def test_cache_hit_restores_follow_up_context(session_client, provider, monkeypatch) -> None:
    """Serving from cache must not lose the video context for follow-ups."""
    _counting_fetch(monkeypatch)
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})
    cached = session_client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_json()

    response = session_client.post("/api/chat", json={"message": "expand on point two"})

    assert response.get_json()["new_conversation"] is False

    # The chat used for the cached request was seeded with the original exchange,
    # so the model sees the transcript and the summary it would have produced.
    seeded = provider.chats[-1]
    assert [role for role, _ in seeded.history[:2]] == ["user", "assistant"]
    assert "<transcript>" in seeded.history[0][1]
    assert seeded.history[1][1] == cached["summary"]


def test_force_refresh_bypasses_both_caches(session_client, provider, monkeypatch) -> None:
    calls = _counting_fetch(monkeypatch)
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    forced = session_client.post(
        "/api/process_video", json={"video_url": VIDEO_URL, "force_refresh": True}
    ).get_json()

    assert forced["cache"] == {"transcript": "miss", "summary": "miss"}
    assert len(calls) == 2
    assert _model_calls(provider) == 2


def test_transcript_failure_is_cached_briefly(session_client, monkeypatch) -> None:
    """Repeated clicks on a blocked video must not hammer the provider."""
    calls: list[str] = []

    def boom(url, **kwargs):
        calls.append(url)
        raise TranscriptFetchError("network down")

    monkeypatch.setattr(app_module, "fetch_transcript", boom)

    first = session_client.post("/api/process_video", json={"video_url": VIDEO_URL})
    second = session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    assert first.status_code == 502
    assert second.status_code == 502
    assert second.get_json()["code"] == "transcript_fetch_failed"
    assert len(calls) == 1  # second response came from the failure cache


def test_force_refresh_retries_after_a_cached_failure(session_client, monkeypatch) -> None:
    calls: list[str] = []

    def boom(url, **kwargs):
        calls.append(url)
        raise TranscriptFetchError("network down")

    monkeypatch.setattr(app_module, "fetch_transcript", boom)
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    forced = session_client.post(
        "/api/process_video", json={"video_url": VIDEO_URL, "force_refresh": True}
    )

    assert forced.status_code == 502
    assert len(calls) == 2  # the retry actually happened


def test_success_clears_a_stale_failure(session_client, provider, monkeypatch) -> None:
    """After a failure expires, a success must overwrite it."""
    def boom(url, **kwargs):
        raise TranscriptFetchError("network down")

    monkeypatch.setattr(app_module, "fetch_transcript", boom)
    session_client.post("/api/process_video", json={"video_url": VIDEO_URL})

    _counting_fetch(monkeypatch)
    ok = session_client.post(
        "/api/process_video", json={"video_url": VIDEO_URL, "force_refresh": True}
    )

    assert ok.status_code == 200
    ext = session_client.application.extensions["yt_summarizer"]
    assert ext["cache"].get_failure("dQw4w9WgXcQ") is None


def test_cache_stores_full_text_while_the_response_is_truncated(
    provider, monkeypatch
) -> None:
    """Truncation is a per-request decision; the cache keeps everything."""
    app = app_module.create_app(make_settings(max_transcript_chars=1_000))
    app.config.update(TESTING=True)
    client = app.test_client()
    client.get("/")

    long_text = "This is a sentence. " * 500  # 10_000 chars
    _counting_fetch(monkeypatch, text=long_text)

    payload = client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_json()

    assert payload["truncated"] is True
    assert payload["transcript_chars"] == len(long_text)

    cached = app.extensions["yt_summarizer"]["cache"].get_transcript("dQw4w9WgXcQ")
    assert cached is not None
    assert len(cached.text) == len(long_text)  # full text survived


def test_healthz_reports_cache_stats(client) -> None:
    payload = client.get("/healthz").get_json()

    assert payload["cache"]["enabled"] == 1
    assert payload["cache"]["transcripts"] == 0


def test_cache_can_be_disabled(provider, monkeypatch) -> None:
    app = app_module.create_app(make_settings(cache_enabled=False))
    app.config.update(TESTING=True)
    client = app.test_client()
    client.get("/")

    calls = _counting_fetch(monkeypatch)
    client.post("/api/process_video", json={"video_url": VIDEO_URL})
    second = client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_json()

    assert second["cache"] == {"transcript": "miss", "summary": "miss"}
    assert len(calls) == 2  # nothing was reused


# --------------------------------------------------------------------------- #
# Context-window safety
# --------------------------------------------------------------------------- #


def test_small_context_model_truncates_a_modest_transcript(provider, monkeypatch) -> None:
    """An 8k-context local model must not be handed a 60k-character transcript."""
    app = app_module.create_app(with_context(make_settings(max_transcript_chars=400_000), 8_192))
    app.config.update(TESTING=True)
    client = app.test_client()
    client.get("/")

    _counting_fetch(monkeypatch, text="слово " * 10_000)  # 60k chars, Cyrillic

    payload = client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_json()

    assert payload["truncated"] is True
    assert payload["context_tokens"] == 8_192
    assert payload["estimated_tokens"] > 8_192  # the original was over budget
    assert payload["transcript_chars"] == 60_000


def test_large_context_model_takes_the_whole_transcript(provider, monkeypatch) -> None:
    app = app_module.create_app(make_settings(max_transcript_chars=400_000))
    app.config.update(TESTING=True)
    client = app.test_client()
    client.get("/")

    _counting_fetch(monkeypatch, text="слово " * 10_000)

    payload = client.post("/api/process_video", json={"video_url": VIDEO_URL}).get_json()

    assert payload["truncated"] is False
    assert payload["context_tokens"] == 1_048_576


def test_healthz_reports_the_context_budget(client) -> None:
    assert client.get("/healthz").get_json()["llm"]["context_tokens"] == 1_048_576


def test_create_app_configures_logging_for_wsgi_servers(monkeypatch) -> None:
    """gunicorn never calls main(), so create_app must set up logging itself."""
    import logging

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    app_module.create_app(make_settings())

    assert root.handlers, "create_app should configure a root handler"


def test_create_app_respects_existing_logging(monkeypatch) -> None:
    """If the host already configured logging, leave it alone."""
    import logging

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    try:
        app_module.create_app(make_settings())
        assert root.handlers == [sentinel]
    finally:
        root.removeHandler(sentinel)
