"""Environment-driven configuration.

Everything the app needs to boot is resolved here, exactly once, so that a
misconfiguration fails loudly at startup instead of in the middle of a request.

The LLM backend is selected entirely by environment variables::

    LLM_PROVIDER=deepseek          # see PROVIDERS below
    LLM_API_KEY=sk-...             # or a provider-specific name (DEEPSEEK_API_KEY, ...)
    LLM_MODEL=deepseek-chat        # optional override
    LLM_BASE_URL=https://...       # optional override (required for "custom")

Switching providers is meant to be a one-line change to ``LLM_PROVIDER``.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger(__name__)

# Load .env next to this file before anything reads os.environ.
load_dotenv(BASE_DIR / ".env")

DEFAULT_PROVIDER = "deepseek"


class ConfigError(RuntimeError):
    """The environment cannot produce a usable configuration."""


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of a supported LLM backend."""

    label: str
    # Wire format, not a vendor: several vendors share the OpenAI shape.
    kind: str  # "openai" | "gemini" | "anthropic"
    default_model: str
    # Checked in order; LLM_API_KEY is the generic fallback for every provider.
    key_envs: tuple[str, ...]
    base_url: str | None = None
    requires_key: bool = True
    note: str = ""
    # Default context window, in tokens. The transcript is clamped to fit this,
    # so switching provider cannot silently overflow the model.
    context_tokens: int = 128_000


PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        label="DeepSeek",
        kind="openai",
        default_model="deepseek-chat",
        key_envs=("DEEPSEEK_API_KEY", "LLM_API_KEY"),
        # DeepSeek documents both https://api.deepseek.com and .../v1.
        base_url="https://api.deepseek.com",
        note="Other models: deepseek-reasoner",
        context_tokens=1_048_576,
    ),
    "openai": ProviderSpec(
        label="OpenAI",
        kind="openai",
        default_model="gpt-4o-mini",
        key_envs=("OPENAI_API_KEY", "LLM_API_KEY"),
        base_url="https://api.openai.com/v1",
        context_tokens=128_000,
    ),
    "gemini": ProviderSpec(
        label="Google Gemini",
        kind="gemini",
        default_model="gemini-2.5-flash",
        key_envs=("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta",
        context_tokens=1_048_576,
    ),
    "anthropic": ProviderSpec(
        label="Anthropic Claude",
        kind="anthropic",
        default_model="claude-sonnet-4-5",
        key_envs=("ANTHROPIC_API_KEY", "LLM_API_KEY"),
        base_url="https://api.anthropic.com/v1",
        context_tokens=200_000,
    ),
    "openrouter": ProviderSpec(
        label="OpenRouter",
        kind="openai",
        default_model="deepseek/deepseek-chat",
        key_envs=("OPENROUTER_API_KEY", "LLM_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        context_tokens=128_000,
    ),
    "groq": ProviderSpec(
        label="Groq",
        kind="openai",
        default_model="llama-3.3-70b-versatile",
        key_envs=("GROQ_API_KEY", "LLM_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        context_tokens=131_072,
    ),
    "ollama": ProviderSpec(
        label="Ollama (local)",
        kind="openai",
        default_model="llama3.2",
        key_envs=("OLLAMA_API_KEY", "LLM_API_KEY"),
        base_url="http://localhost:11434/v1",
        requires_key=False,
        note="Local models are often 8k context; raise LLM_CONTEXT_TOKENS if yours is bigger.",
        context_tokens=8_192,
    ),
    "custom": ProviderSpec(
        label="Custom OpenAI-compatible endpoint",
        kind="openai",
        default_model="",
        key_envs=("LLM_API_KEY",),
        base_url=None,
        requires_key=False,
        note="Requires LLM_BASE_URL and LLM_MODEL.",
    ),
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    kind: str
    label: str
    model: str
    api_key: str
    base_url: str | None
    timeout: float
    max_retries: int
    max_tokens: int
    context_tokens: int
    extra_headers: dict[str, str]

    def describe(self) -> str:
        """Human-readable summary that never includes the key itself."""
        key_state = "set" if self.api_key else "missing"
        return f"{self.label} ({self.provider}) model={self.model!r} key={key_state}"


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    secret_key: str
    session_ttl: int
    session_max: int
    max_transcript_chars: int
    transcript_languages: tuple[str, ...]
    http_timeout: float
    rate_limit_per_minute: int
    cache_enabled: bool
    cache_db: Path
    transcript_cache_ttl: int
    summary_cache_ttl: int
    failure_cache_ttl: int
    host: str
    port: int
    debug: bool
    cookie_secure: bool
    clear_ytdlp_cache: bool
    cookies_file: Path


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = _env_str(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = _env_str(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env_str(name)
    if not raw:
        return default
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    return items or default


def _env_headers(name: str) -> dict[str, str]:
    """Parse a JSON object of extra HTTP headers, e.g. for gateway auth."""
    raw = _env_str(name)
    if not raw:
        return {}
    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must be a JSON object, e.g. '{{\"X-Title\": \"x\"}}'.") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{name} must be a JSON object, got {type(parsed).__name__}.")
    return {str(k): str(v) for k, v in parsed.items()}


def resolve_provider(name: str) -> tuple[str, ProviderSpec]:
    """Map LLM_PROVIDER (plus overrides) onto a ProviderSpec."""
    key = (name or DEFAULT_PROVIDER).strip().lower()

    if key in PROVIDERS:
        return key, PROVIDERS[key]

    # An unknown name is only valid if the user also pointed us at an endpoint.
    if _env_str("LLM_BASE_URL") and _env_str("LLM_MODEL"):
        log.info("Unknown LLM_PROVIDER=%r; treating it as a custom endpoint.", key)
        return key, PROVIDERS["custom"]

    known = ", ".join(sorted(PROVIDERS))
    raise ConfigError(
        f"Unknown LLM_PROVIDER={key!r}. Known providers: {known}. "
        "To use something else, set LLM_PROVIDER=custom with LLM_BASE_URL and LLM_MODEL."
    )


def _resolve_api_key(provider: str, spec: ProviderSpec) -> str:
    for env_name in spec.key_envs:
        value = _env_str(env_name)
        if value:
            return value
    if spec.requires_key:
        expected = " or ".join(spec.key_envs)
        raise ConfigError(
            f"No API key found for provider {provider!r}. Set {expected} in .env "
            f"(see .env.example)."
        )
    return ""


def load_llm_settings() -> LLMSettings:
    provider, spec = resolve_provider(_env_str("LLM_PROVIDER", DEFAULT_PROVIDER) or DEFAULT_PROVIDER)

    model = _env_str("LLM_MODEL") or spec.default_model
    base_url = _env_str("LLM_BASE_URL") or spec.base_url

    if not model:
        raise ConfigError(
            f"Provider {provider!r} has no default model; set LLM_MODEL in .env."
        )
    if not base_url:
        raise ConfigError(
            f"Provider {provider!r} has no endpoint; set LLM_BASE_URL in .env."
        )

    return LLMSettings(
        provider=provider,
        kind=spec.kind,
        label=spec.label,
        model=model,
        api_key=_resolve_api_key(provider, spec),
        base_url=base_url.rstrip("/"),
        timeout=_env_float("LLM_TIMEOUT", 120.0, minimum=1.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 3, minimum=0),
        max_tokens=_env_int("LLM_MAX_TOKENS", 4096, minimum=1),
        context_tokens=_env_int(
            "LLM_CONTEXT_TOKENS", spec.context_tokens, minimum=1_000
        ),
        extra_headers=_env_headers("LLM_EXTRA_HEADERS"),
    )


def _resolve_secret_key() -> str:
    configured = _env_str("SECRET_KEY")
    if configured:
        return configured
    # Ephemeral keys keep local development frictionless, but sessions will not
    # survive a restart and will not work across multiple worker processes.
    log.warning(
        "SECRET_KEY is not set; using an ephemeral key. Sessions will be "
        "invalidated on restart and will break under multiple workers."
    )
    return secrets.token_hex(32)


def load_settings() -> Settings:
    """Build the full application configuration, or raise ConfigError."""
    secret_key = _resolve_secret_key()

    return Settings(
        llm=load_llm_settings(),
        secret_key=secret_key,
        session_ttl=_env_int("SESSION_TTL_SECONDS", 3600, minimum=60),
        session_max=_env_int("SESSION_MAX", 256, minimum=1),
        max_transcript_chars=_env_int("MAX_TRANSCRIPT_CHARS", 400_000, minimum=1_000),
        transcript_languages=_env_list("TRANSCRIPT_LANGUAGES", ("en", "ru", "fr")),
        http_timeout=_env_float("HTTP_TIMEOUT", 20.0, minimum=1.0),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 20, minimum=1),
        cache_enabled=_env_bool("CACHE_ENABLED", True),
        cache_db=BASE_DIR / (_env_str("CACHE_DB", "cache.sqlite3") or "cache.sqlite3"),
        # Transcripts never really change, so they are kept for a long time.
        transcript_cache_ttl=_env_int("TRANSCRIPT_CACHE_TTL", 90 * 24 * 3600, minimum=0),
        summary_cache_ttl=_env_int("SUMMARY_CACHE_TTL", 90 * 24 * 3600, minimum=0),
        # Failures are cached only briefly, to stop us hammering a blocking provider.
        failure_cache_ttl=_env_int("FAILURE_CACHE_TTL", 600, minimum=0),
        host=_env_str("HOST", "127.0.0.1") or "127.0.0.1",
        port=_env_int("PORT", 5001, minimum=1),
        # Debug defaults to off: the Werkzeug debugger is remote code execution
        # if the port is ever reachable.
        debug=_env_bool("DEBUG", False),
        # Set to true when the app is served over HTTPS.
        cookie_secure=_env_bool("SESSION_COOKIE_SECURE", False),
        clear_ytdlp_cache=_env_bool("YTDLP_CLEAR_CACHE", True),
        cookies_file=BASE_DIR / (_env_str("COOKIES_FILE", "cookies.txt") or "cookies.txt"),
    )


def configure_logging(level: str | int = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
