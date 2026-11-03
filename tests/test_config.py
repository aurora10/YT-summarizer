"""Tests for environment-driven configuration and provider resolution."""

from __future__ import annotations

import pytest

from config import ConfigError, load_llm_settings, load_settings, resolve_provider

# Every variable these tests may set, so each test starts from a known state.
_RELEVANT_ENV = (
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "LLM_MAX_TOKENS",
    "LLM_EXTRA_HEADERS",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "OLLAMA_API_KEY",
    "SECRET_KEY",
    "SESSION_TTL_SECONDS",
    "SESSION_MAX",
    "MAX_TRANSCRIPT_CHARS",
    "TRANSCRIPT_LANGUAGES",
    "HTTP_TIMEOUT",
    "RATE_LIMIT_PER_MINUTE",
    "SESSION_COOKIE_SECURE",
    "CACHE_ENABLED",
    "CACHE_DB",
    "TRANSCRIPT_CACHE_TTL",
    "SUMMARY_CACHE_TTL",
    "FAILURE_CACHE_TTL",
    "DEBUG",
    "HOST",
    "PORT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _RELEVANT_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# --------------------------------------------------------------------------- #
# Provider defaults
# --------------------------------------------------------------------------- #


def test_defaults_to_deepseek(clean_env) -> None:
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    settings = load_llm_settings()

    assert settings.provider == "deepseek"
    assert settings.kind == "openai"
    assert settings.model == "deepseek-chat"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.api_key == "sk-deepseek"


def test_generic_key_works_when_only_provider_is_set(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "deepseek")
    clean_env.setenv("LLM_API_KEY", "sk-generic")

    assert load_llm_settings().api_key == "sk-generic"


def test_provider_specific_key_beats_generic(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-generic")
    clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    clean_env.setenv("LLM_PROVIDER", "openai")

    settings = load_llm_settings()

    assert settings.api_key == "sk-openai"
    assert settings.model == "gpt-4o-mini"
    assert settings.base_url == "https://api.openai.com/v1"


def test_switching_provider_needs_only_the_name(clean_env) -> None:
    """The whole point: same .env, different LLM_PROVIDER, different backend."""
    clean_env.setenv("LLM_API_KEY", "sk-generic")
    clean_env.setenv("LLM_PROVIDER", "anthropic")

    settings = load_llm_settings()

    assert settings.kind == "anthropic"
    assert settings.model == "claude-sonnet-4-5"
    assert settings.base_url == "https://api.anthropic.com/v1"


def test_gemini_accepts_legacy_google_api_key(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "gemini")
    clean_env.setenv("GOOGLE_API_KEY", "AIza-legacy")

    settings = load_llm_settings()

    assert settings.kind == "gemini"
    assert settings.api_key == "AIza-legacy"
    assert settings.model == "gemini-2.5-flash"


def test_ollama_needs_no_key(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "ollama")

    settings = load_llm_settings()

    assert settings.api_key == ""
    assert settings.base_url == "http://localhost:11434/v1"


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #


def test_model_and_base_url_overrides(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "deepseek")
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("LLM_MODEL", "deepseek-reasoner")
    clean_env.setenv("LLM_BASE_URL", "https://proxy.internal/v1/")

    settings = load_llm_settings()

    assert settings.model == "deepseek-reasoner"
    assert settings.base_url == "https://proxy.internal/v1"  # trailing slash trimmed


def test_custom_provider_uses_overrides(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "custom")
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("LLM_BASE_URL", "https://my-gateway/v1")
    clean_env.setenv("LLM_MODEL", "my-model")

    settings = load_llm_settings()

    assert settings.provider == "custom"
    assert settings.model == "my-model"
    assert settings.base_url == "https://my-gateway/v1"


def test_unknown_name_with_endpoint_is_treated_as_custom(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "my-internal-llm")
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("LLM_BASE_URL", "https://internal/v1")
    clean_env.setenv("LLM_MODEL", "m")

    settings = load_llm_settings()

    assert settings.kind == "openai"
    assert settings.base_url == "https://internal/v1"


def test_extra_headers_are_parsed(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("LLM_EXTRA_HEADERS", '{"HTTP-Referer": "https://x.dev"}')

    assert load_llm_settings().extra_headers == {"HTTP-Referer": "https://x.dev"}


def test_tuning_values_have_defaults_and_override(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    defaults = load_llm_settings()
    assert defaults.timeout == 120.0
    assert defaults.max_retries == 3
    assert defaults.max_tokens == 4096

    clean_env.setenv("LLM_TIMEOUT", "30")
    clean_env.setenv("LLM_MAX_RETRIES", "0")
    configured = load_llm_settings()
    assert configured.timeout == 30.0
    assert configured.max_retries == 0


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def test_missing_key_names_the_variables_to_set(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "deepseek")

    with pytest.raises(ConfigError) as excinfo:
        load_llm_settings()

    message = str(excinfo.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "LLM_API_KEY" in message


def test_unknown_provider_lists_alternatives(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "hal9000")
    clean_env.setenv("LLM_API_KEY", "sk-x")

    with pytest.raises(ConfigError) as excinfo:
        load_llm_settings()

    message = str(excinfo.value)
    assert "hal9000" in message
    assert "deepseek" in message
    assert "custom" in message


def test_custom_provider_requires_base_url(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "custom")
    clean_env.setenv("LLM_MODEL", "m")

    with pytest.raises(ConfigError, match="LLM_BASE_URL"):
        load_llm_settings()


def test_custom_provider_requires_model(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "custom")
    clean_env.setenv("LLM_BASE_URL", "https://x/v1")

    with pytest.raises(ConfigError, match="LLM_MODEL"):
        load_llm_settings()


@pytest.mark.parametrize(
    ("var", "value"),
    [("LLM_TIMEOUT", "soon"), ("LLM_MAX_RETRIES", "many"), ("LLM_MAX_TOKENS", "-1"), ("LLM_TIMEOUT", "0")],
)
def test_invalid_numeric_settings_fail_loudly(clean_env, var, value) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv(var, value)

    with pytest.raises(ConfigError):
        load_llm_settings()


def test_malformed_extra_headers_fail_loudly(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("LLM_EXTRA_HEADERS", "not json")

    with pytest.raises(ConfigError, match="LLM_EXTRA_HEADERS"):
        load_llm_settings()


def test_resolve_provider_is_case_insensitive(clean_env) -> None:
    name, spec = resolve_provider("  DeepSeek  ")
    assert name == "deepseek"
    assert spec.label == "DeepSeek"


# --------------------------------------------------------------------------- #
# Full settings
# --------------------------------------------------------------------------- #


def test_load_settings_defaults(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")

    settings = load_settings()

    assert settings.session_ttl == 3600
    assert settings.session_max == 256
    assert settings.max_transcript_chars == 400_000
    assert settings.transcript_languages == ("en", "ru", "fr")
    assert settings.rate_limit_per_minute == 20
    # Safe defaults: no debugger, not bound to a public interface.
    assert settings.debug is False
    assert settings.host == "127.0.0.1"
    assert settings.cookie_secure is False


def test_debug_requires_opt_in(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    assert load_settings().debug is False

    clean_env.setenv("DEBUG", "1")
    assert load_settings().debug is True


def test_secret_key_is_generated_when_absent(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")

    first = load_settings().secret_key
    second = load_settings().secret_key

    assert first  # non-empty
    assert first != second  # ephemeral, hence the startup warning


def test_secret_key_prefers_the_configured_value(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("SECRET_KEY", "stable-key")

    assert load_settings().secret_key == "stable-key"


def test_transcript_languages_override(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("TRANSCRIPT_LANGUAGES", "de, en ,,fr")

    assert load_settings().transcript_languages == ("de", "en", "fr")


def test_describe_never_leaks_the_key(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-super-secret")

    description = load_llm_settings().describe()

    assert "super-secret" not in description
    assert "set" in description


# --------------------------------------------------------------------------- #
# Cache settings
# --------------------------------------------------------------------------- #


def test_cache_defaults(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")

    settings = load_settings()

    assert settings.cache_enabled is True
    assert settings.cache_db.name == "cache.sqlite3"
    assert settings.transcript_cache_ttl == 90 * 24 * 3600
    assert settings.summary_cache_ttl == 90 * 24 * 3600
    # Failures are remembered only briefly.
    assert settings.failure_cache_ttl == 600


def test_cache_can_be_disabled_and_relocated(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("CACHE_ENABLED", "false")
    clean_env.setenv("CACHE_DB", "data/my-cache.sqlite3")

    settings = load_settings()

    assert settings.cache_enabled is False
    assert settings.cache_db.name == "my-cache.sqlite3"
    assert settings.cache_db.parent.name == "data"


def test_cache_ttls_accept_zero_for_never_expire(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("TRANSCRIPT_CACHE_TTL", "0")

    assert load_settings().transcript_cache_ttl == 0


def test_invalid_cache_ttl_fails_loudly(clean_env) -> None:
    clean_env.setenv("LLM_API_KEY", "sk-x")
    clean_env.setenv("FAILURE_CACHE_TTL", "forever")

    with pytest.raises(ConfigError):
        load_settings()
