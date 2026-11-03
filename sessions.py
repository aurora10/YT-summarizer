"""Bounded, expiring in-memory session store.

Holds conversation state only. Transcript text used to live here too, which meant
a full copy per session and no reuse across restarts; it now lives in the durable
cache (``cache.py``).

The previous implementation kept every session forever in a plain dict. This
store evicts expired entries and enforces a hard capacity ceiling.

Single-process only by design: with multiple worker processes each worker would
hold its own copy. For a multi-worker deployment, swap this for Redis behind the
same interface.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from llm import Chat


@dataclass
class SessionData:
    """Everything we remember about one browser session."""

    chat: Chat | None = None
    video_url: str | None = None
    video_id: str | None = None
    lang_code: str | None = None
    truncated: bool = False
    last_seen: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        self.chat = None
        self.video_url = None
        self.video_id = None
        self.lang_code = None
        self.truncated = False


class SessionStore:
    """LRU + TTL store keyed by session user id."""

    def __init__(
        self,
        ttl_seconds: int,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._data: OrderedDict[str, SessionData] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._data)

    def get(self, user_id: str) -> SessionData:
        """Fetch (or create) the session for ``user_id``, refreshing its TTL."""
        if not user_id:
            raise ValueError("user_id must be a non-empty string")

        with self._lock:
            now = self._clock()

            existing = self._data.get(user_id)
            if existing is not None and not self._is_expired(existing, now):
                existing.last_seen = now
                self._data.move_to_end(user_id)
                return existing

            # Missing or expired: start clean rather than resurrecting stale data.
            data = SessionData(last_seen=now)
            self._data[user_id] = data
            self._data.move_to_end(user_id)
            self._enforce_capacity()
            return data

    def clear(self, user_id: str) -> bool:
        """Drop the session entirely. Returns True if one existed."""
        with self._lock:
            existing = self._data.pop(user_id, None)
            return existing is not None

    def reset(self, user_id: str) -> SessionData:
        """Clear the conversation but keep the session slot."""
        data = self.get(user_id)
        with self._lock:
            data.reset()
        return data

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._evict_expired()
            return {"sessions": len(self._data), "capacity": self._max, "ttl": self._ttl}

    # -- internals ---------------------------------------------------------- #

    def _is_expired(self, data: SessionData, now: float) -> bool:
        return (now - data.last_seen) > self._ttl

    def _evict_expired(self) -> None:
        now = self._clock()
        stale = [uid for uid, data in self._data.items() if self._is_expired(data, now)]
        for uid in stale:
            self._data.pop(uid, None)

    def _enforce_capacity(self) -> None:
        self._evict_expired()
        while len(self._data) > self._max:
            evicted_id, _ = self._data.popitem(last=False)  # oldest first
            self._log_eviction(evicted_id)

    @staticmethod
    def _log_eviction(user_id: str) -> None:
        import logging

        logging.getLogger(__name__).debug("Evicted session %s", user_id)


class RateLimiter:
    """Tiny per-key sliding-window limiter to protect the LLM quota.

    Per-process and in-memory; adequate for a single-user tool, and honest about
    being a speed bump rather than a real quota system.
    """

    def __init__(
        self,
        max_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        window: float = 60.0,
    ) -> None:
        self._max = max_per_minute
        self._window = window
        self._clock = clock
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            hits = [t for t in self._hits.get(key, ()) if now - t < self._window]
            if len(hits) >= self._max:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    def prune(self) -> None:
        """Drop bookkeeping for keys with no recent activity."""
        now = self._clock()
        with self._lock:
            for key in [k for k, v in self._hits.items() if not any(now - t < self._window for t in v)]:
                self._hits.pop(key, None)
