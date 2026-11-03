"""Tests for the durable transcript/summary/failure cache."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cache import CachedSummary, NullCache, SummaryCache, build_cache


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(clock) -> SummaryCache:
    return SummaryCache(":memory:", clock=clock)


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


def test_transcript_roundtrip(cache) -> None:
    cache.put_transcript(
        "dQw4w9WgXcQ", text="hello world", lang_code="en", source="youtube-api"
    )

    hit = cache.get_transcript("dQw4w9WgXcQ")

    assert hit is not None
    assert hit.text == "hello world"
    assert hit.lang_code == "en"
    assert hit.source == "youtube-api"


def test_transcript_miss_returns_none(cache) -> None:
    assert cache.get_transcript("AAAAAAAAAAA") is None


def test_transcript_expires(cache, clock) -> None:
    cache.put_transcript("vid", text="x", lang_code="en", source="s")
    assert cache.get_transcript("vid") is not None

    clock.advance(91 * 24 * 3600)  # past the 90-day default

    assert cache.get_transcript("vid") is None


def test_ttl_of_zero_means_never_expire(clock) -> None:
    cache = SummaryCache(":memory:", transcript_ttl=0, clock=clock)
    cache.put_transcript("vid", text="x", lang_code="en", source="s")

    clock.advance(100 * 365 * 24 * 3600)

    assert cache.get_transcript("vid") is not None


def test_transcript_is_overwritten_on_refetch(cache) -> None:
    cache.put_transcript("vid", text="old", lang_code="en", source="s")
    cache.put_transcript("vid", text="new", lang_code="en", source="s")

    assert cache.get_transcript("vid").text == "new"  # type: ignore[union-attr]


def test_long_transcript_is_stored_in_full(cache) -> None:
    """The cache must keep everything; truncation is a per-request decision."""
    text = "x" * 500_000
    cache.put_transcript("vid", text=text, lang_code="en", source="s")

    assert len(cache.get_transcript("vid").text) == 500_000  # type: ignore[union-attr]


def test_unicode_transcript_roundtrip(cache) -> None:
    text = "Привет, мир! «Кавычки» и эмодзи 🐘"
    cache.put_transcript("vid", text=text, lang_code="ru", source="yt-dlp")

    assert cache.get_transcript("vid").text == text  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def test_summary_roundtrip(cache) -> None:
    key = cache.summary_key("vid", model="deepseek-chat", prompt_version=1, max_chars=1000)
    cache.put_summary(
        key, video_id="vid", summary="a summary", truncated=False, transcript_chars=42
    )

    hit = cache.get_summary(key)

    assert hit == CachedSummary(summary="a summary", truncated=False, transcript_chars=42)


def test_summary_key_changes_with_model(cache) -> None:
    a = cache.summary_key("vid", model="deepseek-chat", prompt_version=1, max_chars=1000)
    b = cache.summary_key("vid", model="gpt-4o-mini", prompt_version=1, max_chars=1000)
    assert a != b


def test_summary_key_changes_with_prompt_version(cache) -> None:
    """Editing the prompt must not keep serving summaries from the old one."""
    a = cache.summary_key("vid", model="m", prompt_version=1, max_chars=1000)
    b = cache.summary_key("vid", model="m", prompt_version=2, max_chars=1000)
    assert a != b


def test_summary_key_changes_with_max_chars(cache) -> None:
    a = cache.summary_key("vid", model="m", prompt_version=1, max_chars=1000)
    b = cache.summary_key("vid", model="m", prompt_version=1, max_chars=2000)
    assert a != b


def test_summary_key_changes_with_language(cache) -> None:
    a = cache.summary_key("vid", model="m", prompt_version=1, max_chars=1, lang_code="en")
    b = cache.summary_key("vid", model="m", prompt_version=1, max_chars=1, lang_code="ru")
    assert a != b


def test_summary_key_is_stable(cache) -> None:
    kwargs = dict(model="m", prompt_version=1, max_chars=1, lang_code="en")
    assert cache.summary_key("vid", **kwargs) == cache.summary_key("vid", **kwargs)


def test_summary_expires(cache, clock) -> None:
    key = cache.summary_key("vid", model="m", prompt_version=1, max_chars=1)
    cache.put_summary(key, video_id="vid", summary="s", truncated=False, transcript_chars=1)

    clock.advance(91 * 24 * 3600)

    assert cache.get_summary(key) is None


# --------------------------------------------------------------------------- #
# Failures (negative cache)
# --------------------------------------------------------------------------- #


def test_failure_roundtrip(cache) -> None:
    cache.put_failure("vid", status=502, code="transcript_fetch_failed", message="blocked")

    hit = cache.get_failure("vid")

    assert hit is not None
    assert (hit.status, hit.code, hit.message) == (502, "transcript_fetch_failed", "blocked")


def test_failure_expires_quickly(cache, clock) -> None:
    """A transient block must not be baked in for days."""
    cache.put_failure("vid", status=502, code="blocked", message="m")

    clock.advance(601)  # default failure TTL is 600s

    assert cache.get_failure("vid") is None


def test_clear_failure_after_success(cache) -> None:
    cache.put_failure("vid", status=502, code="blocked", message="m")
    cache.clear_failure("vid")

    assert cache.get_failure("vid") is None


def test_clear_failure_is_idempotent(cache) -> None:
    cache.clear_failure("nobody")  # must not raise


# --------------------------------------------------------------------------- #
# Persistence and maintenance
# --------------------------------------------------------------------------- #


def test_items_survive_a_restart(tmp_path) -> None:
    """A new instance on the same file must see the previous data."""
    db = tmp_path / "cache.sqlite3"

    first = SummaryCache(db)
    first.put_transcript("vid", text="persisted", lang_code="en", source="s")
    key = first.summary_key("vid", model="m", prompt_version=1, max_chars=1)
    first.put_summary(key, video_id="vid", summary="sum", truncated=False, transcript_chars=1)
    first.close()

    second = SummaryCache(db)
    assert second.get_transcript("vid").text == "persisted"  # type: ignore[union-attr]
    assert second.get_summary(key).summary == "sum"  # type: ignore[union-attr]
    second.close()


def test_purge_removes_expired_rows(clock) -> None:
    cache = SummaryCache(":memory:", clock=clock)
    cache.put_transcript("a", text="x", lang_code="en", source="s")
    cache.put_failure("b", status=422, code="c", message="m")

    clock.advance(700)  # failures expire at 600s, transcripts at 90 days
    removed = cache.purge_expired()

    assert removed == 1
    assert cache.stats()["failures"] == 0
    assert cache.stats()["transcripts"] == 1


def test_stats_reports_counts(cache) -> None:
    cache.put_transcript("a", text="x", lang_code="en", source="s")
    cache.put_failure("b", status=422, code="c", message="m")

    stats = cache.stats()

    assert stats == {"enabled": 1, "transcripts": 1, "summaries": 0, "failures": 1}


def test_unwritable_location_degrades_to_unavailable(tmp_path) -> None:
    """A broken cache must not break the app."""
    # A directory cannot be opened as a database file.
    cache = SummaryCache(tmp_path)

    assert cache.available is False
    assert cache.get_transcript("vid") is None
    cache.put_transcript("vid", text="x", lang_code="en", source="s")  # must not raise
    assert cache.stats()["transcripts"] == 0


def test_concurrent_access(tmp_path) -> None:
    """gunicorn runs threads; the cache must cope with concurrent writers."""
    cache = SummaryCache(tmp_path / "cache.sqlite3")
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            vid = f"video{index:03d}"
            cache.put_transcript(vid, text=f"text{index}", lang_code="en", source="s")
            key = cache.summary_key(vid, model="m", prompt_version=1, max_chars=1)
            cache.put_summary(
                key, video_id=vid, summary="s", truncated=False, transcript_chars=1
            )
            assert cache.get_transcript(vid) is not None
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert cache.stats()["transcripts"] == 20
    cache.close()


# --------------------------------------------------------------------------- #
# NullCache / factory
# --------------------------------------------------------------------------- #


def test_disabled_cache_is_a_noop(tmp_path) -> None:
    cache = build_cache(tmp_path / "cache.sqlite3", enabled=False)

    assert isinstance(cache, NullCache)
    cache.put_transcript("vid", text="x", lang_code="en", source="s")
    assert cache.get_transcript("vid") is None
    assert cache.stats() == {"enabled": 0}
    assert not (tmp_path / "cache.sqlite3").exists()


def test_enabled_cache_is_a_real_one(tmp_path) -> None:
    cache = build_cache(tmp_path / "cache.sqlite3", enabled=True)

    assert isinstance(cache, SummaryCache)
    assert cache.available is True
    cache.close()


def test_creates_parent_directory(tmp_path) -> None:
    nested = tmp_path / "data" / "nested" / "cache.sqlite3"
    cache = SummaryCache(nested)

    assert cache.available is True
    assert Path(nested).exists()
    cache.close()
