"""Tests for the bounded session store and the rate limiter."""

from __future__ import annotations

import pytest

from sessions import RateLimiter, SessionStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------- #
# SessionStore
# --------------------------------------------------------------------------- #


def test_get_creates_empty_session() -> None:
    store = SessionStore(ttl_seconds=60, max_entries=10)
    data = store.get("user-1")

    assert data.chat is None
    assert data.video_id is None
    assert len(store) == 1


def test_get_returns_same_object_for_active_session() -> None:
    store = SessionStore(ttl_seconds=60, max_entries=10)
    first = store.get("user-1")
    first.video_id = "dQw4w9WgXcQ"

    assert store.get("user-1") is first
    assert store.get("user-1").video_id == "dQw4w9WgXcQ"


def test_expired_session_is_recreated_empty() -> None:
    clock = FakeClock()
    store = SessionStore(ttl_seconds=60, max_entries=10, clock=clock)

    data = store.get("user-1")
    data.video_id = "dQw4w9WgXcQ"

    clock.advance(61)
    fresh = store.get("user-1")

    assert fresh is not data
    assert fresh.video_id is None


def test_activity_refreshes_ttl() -> None:
    clock = FakeClock()
    store = SessionStore(ttl_seconds=60, max_entries=10, clock=clock)

    data = store.get("user-1")
    data.video_id = "keepMe00000"

    # Touch it every 30s for 3 minutes: it should never expire.
    for _ in range(6):
        clock.advance(30)
        assert store.get("user-1") is data

    assert store.get("user-1").video_id == "keepMe00000"


def test_capacity_is_enforced_by_evicting_oldest() -> None:
    store = SessionStore(ttl_seconds=600, max_entries=3)

    for index in range(3):
        store.get(f"user-{index}")
    assert len(store) == 3

    store.get("user-3")  # over capacity

    assert len(store) == 3
    # Oldest (user-0) is gone; the newcomer survived.
    assert store.get("user-3").chat is None
    assert store.clear("user-0") is False
    assert store.clear("user-2") is True


def test_expired_entries_are_dropped_on_read() -> None:
    clock = FakeClock()
    store = SessionStore(ttl_seconds=10, max_entries=10, clock=clock)

    store.get("a")
    store.get("b")
    clock.advance(11)

    assert len(store) == 0


def test_clear_reports_presence() -> None:
    store = SessionStore(ttl_seconds=60, max_entries=10)
    store.get("user-1")

    assert store.clear("user-1") is True
    assert store.clear("user-1") is False


def test_reset_clears_conversation_but_keeps_slot() -> None:
    store = SessionStore(ttl_seconds=60, max_entries=10)

    data = store.get("user-1")
    data.video_id = "dQw4w9WgXcQ"
    data.video_id = "dQw4w9WgXcQ"

    store.reset("user-1")
    again = store.get("user-1")

    assert again is data
    assert again.video_id is None
    assert again.video_id is None


def test_stats_reports_capacity() -> None:
    store = SessionStore(ttl_seconds=60, max_entries=5)
    store.get("a")

    stats = store.stats()
    assert stats["sessions"] == 1
    assert stats["capacity"] == 5
    assert stats["ttl"] == 60


@pytest.mark.parametrize("ttl,max_entries", [(0, 10), (-1, 10), (60, 0), (60, -5)])
def test_invalid_construction_is_rejected(ttl: int, max_entries: int) -> None:
    with pytest.raises(ValueError):
        SessionStore(ttl_seconds=ttl, max_entries=max_entries)


def test_get_requires_user_id() -> None:
    store = SessionStore(ttl_seconds=60, max_entries=10)
    with pytest.raises(ValueError):
        store.get("")


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #


def test_rate_limiter_allows_up_to_the_limit() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_per_minute=3, clock=clock)

    assert [limiter.allow("u") for _ in range(3)] == [True, True, True]
    assert limiter.allow("u") is False


def test_rate_limiter_window_slides() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_per_minute=2, clock=clock)

    assert limiter.allow("u") is True
    assert limiter.allow("u") is True
    assert limiter.allow("u") is False

    clock.advance(61)
    assert limiter.allow("u") is True


def test_rate_limiter_is_per_key() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_per_minute=1, clock=clock)

    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is True


def test_rate_limiter_prune_drops_idle_keys() -> None:
    clock = FakeClock()
    limiter = RateLimiter(max_per_minute=1, clock=clock)

    limiter.allow("a")
    clock.advance(61)
    limiter.prune()

    assert limiter._hits == {}  # noqa: SLF001 - prune is the behaviour under test
