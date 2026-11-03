"""Provider-agnostic chat abstraction.

Add a provider by:
  1. adding a ``ProviderSpec`` in ``config.PROVIDERS`` (most vendors need nothing
     more, because they speak one of the three wire formats below), or
  2. subclassing ``Provider`` and registering it in ``_KINDS`` for a new format.

No vendor SDKs are required: adapters speak HTTP via ``requests``, which keeps
timeouts, retries and error mapping identical for every backend.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests

from config import LLMSettings

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class LLMError(RuntimeError):
    """Base class for every failure talking to an LLM provider."""


class LLMConfigError(LLMError):
    """The provider is misconfigured (bad key, unknown model, ...)."""


class LLMAuthError(LLMError):
    """The provider rejected our credentials."""


class LLMRateLimitError(LLMError):
    """The provider asked us to slow down."""


class LLMTimeoutError(LLMError):
    """The provider did not answer in time."""


class LLMContentBlockedError(LLMError):
    """The provider refused to process the content on policy grounds."""


class LLMResponseError(LLMError):
    """The provider answered, but not with anything we can use."""


# Retryable: transient by nature. Everything else fails fast.
_RETRYABLE = (LLMRateLimitError, LLMTimeoutError)

# Providers disagree on how they report bad credentials. Google, for example,
# returns HTTP 400 with an "API key not valid" message rather than a 401.
_AUTH_HINTS = (
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "unauthenticated",
    "permission denied",
    "authentication fail",
)


# --------------------------------------------------------------------------- #
# Conversation primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Chat:
    """A single conversation with one provider.

    This is the interface the rest of the app codes against::

        chat = provider.start_chat(system_instruction="...")
        reply = chat.send("...")
        for turn in chat.history: ...
    """

    def __init__(
        self,
        provider: Provider,
        system_instruction: str = "",
        history: Iterable[Turn] | None = None,
    ) -> None:
        self._provider = provider
        self._system_instruction = system_instruction
        self._history: list[Turn] = list(history or ())

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def system_instruction(self) -> str:
        return self._system_instruction

    @property
    def history(self) -> list[Turn]:
        return list(self._history)

    @property
    def is_empty(self) -> bool:
        return not self._history

    def send(self, message: str) -> str:
        """Append ``message``, get a reply, append the reply, return its text."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        self._history.append(Turn("user", message))
        try:
            reply = self._provider.complete(self._system_instruction, self._history)
        except BaseException:
            # A failed turn must not poison the history we replay next time.
            self._history.pop()
            raise

        self._history.append(Turn("assistant", reply))
        return reply


# --------------------------------------------------------------------------- #
# Provider base
# --------------------------------------------------------------------------- #


@dataclass
class Provider:
    """Transport-agnostic base class: retries, timeouts, error mapping."""

    settings: LLMSettings
    session: requests.Session = field(init=False, repr=False)

    kind = "base"

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "yt-summarizer/1.0"})
        self.session.headers.update(self.settings.extra_headers)

    # -- public API --------------------------------------------------------- #

    def start_chat(self, system_instruction: str = "") -> Chat:
        return Chat(self, system_instruction)

    def complete(self, system_instruction: str, turns: Sequence[Turn]) -> str:
        """Run one completion, retrying transient failures with backoff."""
        attempts = self.settings.max_retries + 1
        last_error: LLMError | None = None

        for attempt in range(attempts):
            try:
                return self._complete(system_instruction, turns)
            except _RETRYABLE as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                # Exponential backoff with jitter so parallel users don't sync up.
                delay = min(2.0**attempt, 8.0) * (0.5 + random.random())
                log.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    self.settings.label,
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    def close(self) -> None:
        self.session.close()

    # -- to implement ------------------------------------------------------- #

    def _complete(self, system_instruction: str, turns: Sequence[Turn]) -> str:
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------- #

    def _post(self, url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **headers}
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=(min(10.0, self.settings.timeout), self.settings.timeout),
            )
        except requests.Timeout as exc:
            raise LLMTimeoutError(
                f"{self.settings.label} did not respond within "
                f"{self.settings.timeout:.0f}s."
            ) from exc
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach {self.settings.label}: {exc}") from exc

        if response.status_code >= 400:
            self._raise_for_error(response)

        try:
            return response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"{self.settings.label} returned a non-JSON response "
                f"({response.status_code})."
            ) from exc

    def _raise_for_error(self, response: requests.Response) -> None:
        label = self.settings.label
        status = response.status_code
        detail = _error_detail(response)

        # Not every provider uses 401/403 for bad credentials: Google returns
        # 400 with a specific reason string for an invalid key.
        lowered = detail.lower()
        looks_like_auth = any(hint in lowered for hint in _AUTH_HINTS)

        if status in (401, 403, 402) or looks_like_auth:
            raise LLMAuthError(
                f"{label} rejected the request's credentials ({status}){detail}. "
                f"Check the API key (and account credit) for "
                f"LLM_PROVIDER={self.settings.provider!r} in .env."
            )
        if status == 429:
            raise LLMRateLimitError(f"{label} rate limit or quota exhausted ({status}){detail}.")
        if status >= 500:
            # Surface as a timeout-class error so the retry loop handles it.
            raise LLMTimeoutError(f"{label} server error ({status}){detail}.")
        raise LLMError(f"{label} request failed ({status}){detail}.")


def _error_detail(response: requests.Response, limit: int = 300) -> str:
    """Best-effort extraction of a provider's error message, safely truncated."""
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip().replace("\n", " ")
        return f": {text[:limit].rstrip('.')}" if text else ""

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
            if message:
                return f": {str(message)[:limit].rstrip('.')}"
        if isinstance(error, str):
            return f": {error[:limit].rstrip('.')}"
        for key in ("message", "detail", "reason"):
            if isinstance(body.get(key), str):
                return f": {body[key][:limit].rstrip('.')}"
    return ""


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


class OpenAICompatibleProvider(Provider):
    """Covers DeepSeek, OpenAI, OpenRouter, Groq, Ollama, vLLM, LM Studio, ...

    All of them expose ``POST {base_url}/chat/completions`` with bearer auth.
    """

    kind = "openai"

    def _complete(self, system_instruction: str, turns: Sequence[Turn]) -> str:
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.extend(turn.as_dict() for turn in turns)

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
        }

        headers = {}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        body = self._post(f"{self.settings.base_url}/chat/completions", headers=headers, payload=payload)

        choices = body.get("choices") or []
        if not choices:
            raise LLMContentBlockedError(
                f"{self.settings.label} returned no completion; the content was "
                "likely filtered by its safety systems."
            )

        choice = choices[0]
        if choice.get("finish_reason") == "content_filter":
            raise LLMContentBlockedError(
                f"{self.settings.label} blocked this content under its safety policy."
            )

        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError(f"{self.settings.label} returned an empty completion.")
        return text.strip()


class GeminiProvider(Provider):
    """Google Generative Language API (``generateContent``)."""

    kind = "gemini"

    _BLOCKED_FINISH_REASONS = {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
    }

    def _complete(self, system_instruction: str, turns: Sequence[Turn]) -> str:
        contents = [
            {
                "role": "model" if turn.role == "assistant" else "user",
                "parts": [{"text": turn.content}],
            }
            for turn in turns
        ]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.settings.max_tokens},
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        body = self._post(
            f"{self.settings.base_url}/models/{self.settings.model}:generateContent",
            headers={"x-goog-api-key": self.settings.api_key},
            payload=payload,
        )

        feedback = body.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LLMContentBlockedError(
                "Gemini blocked the prompt "
                f"({feedback['blockReason']}) under its safety policy."
            )

        candidates = body.get("candidates") or []
        if not candidates:
            raise LLMContentBlockedError("Gemini returned no candidates.")

        candidate = candidates[0]
        finish = (candidate.get("finishReason") or "").upper()
        if finish in self._BLOCKED_FINISH_REASONS:
            raise LLMContentBlockedError(
                f"Gemini stopped generating ({finish}) under its safety policy."
            )

        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise LLMResponseError("Gemini returned an empty response.")
        return text


class AnthropicProvider(Provider):
    """Anthropic Messages API."""

    kind = "anthropic"
    _API_VERSION = "2023-06-01"

    def _complete(self, system_instruction: str, turns: Sequence[Turn]) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "messages": [turn.as_dict() for turn in turns],
        }
        if system_instruction:
            payload["system"] = system_instruction

        body = self._post(
            f"{self.settings.base_url}/messages",
            headers={
                "x-api-key": self.settings.api_key,
                "anthropic-version": self._API_VERSION,
            },
            payload=payload,
        )

        if (body.get("stop_reason") or "") == "refusal":
            raise LLMContentBlockedError(
                "Anthropic declined to process this content under its usage policy."
            )

        blocks = body.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text:
            raise LLMResponseError("Anthropic returned an empty response.")
        return text


_KINDS: dict[str, type[Provider]] = {
    "openai": OpenAICompatibleProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(settings: LLMSettings) -> Provider:
    """Instantiate the adapter matching the configured wire format."""
    try:
        provider_cls = _KINDS[settings.kind]
    except KeyError:
        raise LLMConfigError(
            f"No adapter for wire format {settings.kind!r} "
            f"(provider {settings.provider!r})."
        ) from None

    provider = provider_cls(settings)
    log.info("LLM backend ready: %s", settings.describe())
    return provider
