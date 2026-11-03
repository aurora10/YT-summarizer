"""Durable caches for transcripts, summaries and fetch failures (SQLite).

Two properties make this worth doing:

* A transcript is effectively immutable per video, so one successful fetch should
  serve forever — that is what protects the app from a YouTube block.
* A summary is expensive (tens of thousands of input tokens) yet identical for
  every user, so it is cacheable too.

The cache is best-effort by design: any storage error is logged and treated as a
miss, because a broken cache must never break the request.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    video_id   TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    lang_code  TEXT,
    source     TEXT,
    fetched_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    cache_key        TEXT PRIMARY KEY,
    video_id         TEXT NOT NULL,
    summary          TEXT NOT NULL,
    truncated        INTEGER NOT NULL DEFAULT 0,
    transcript_chars INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS failures (
    video_id   TEXT PRIMARY KEY,
    status     INTEGER NOT NULL,
    code       TEXT NOT NULL,
    message    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summaries_video ON summaries (video_id);
"""


@dataclass(frozen=True)
class CachedTranscript:
    text: str
    lang_code: str | None
    source: str


@dataclass(frozen=True)
class CachedSummary:
    summary: str
    truncated: bool
    transcript_chars: int


@dataclass(frozen=True)
class CachedFailure:
    status: int
    code: str
    message: str


class NullCache:
    """No-op cache used when caching is disabled or the DB is unavailable."""

    available = False

    def get_transcript(self, video_id: str) -> CachedTranscript | None:
        return None

    def put_transcript(self, video_id: str, *, text: str, lang_code: str | None, source: str) -> None:
        return None

    def get_summary(self, cache_key: str) -> CachedSummary | None:
        return None

    def put_summary(self, cache_key: str, **_: object) -> None:
        return None

    def get_failure(self, video_id: str) -> CachedFailure | None:
        return None

    def put_failure(self, video_id: str, **_: object) -> None:
        return None

    def clear_failure(self, video_id: str) -> None:
        return None

    def purge_expired(self) -> int:
        return 0

    def stats(self) -> dict[str, int]:
        return {"enabled": 0}

    def summary_key(self, video_id: str, **_: object) -> str:
        return video_id

    def close(self) -> None:
        return None


class SummaryCache:
    """SQLite-backed cache for transcripts, summaries and recent failures."""

    def __init__(
        self,
        path: str | Path,
        *,
        transcript_ttl: int = 90 * 24 * 3600,
        summary_ttl: int = 90 * 24 * 3600,
        failure_ttl: int = 600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = str(path)
        self._transcript_ttl = transcript_ttl
        self._summary_ttl = summary_ttl
        self._failure_ttl = failure_ttl
        self._clock = clock
        self._lock = threading.RLock()
        self._writes = 0

        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        try:
            self._conn: sqlite3.Connection | None = sqlite3.connect(
                self._path, check_same_thread=False, timeout=5.0
            )
            self._conn.row_factory = sqlite3.Row
            self._configure()
            self._create_schema()
        except sqlite3.Error as exc:  # pragma: no cover - depends on the filesystem
            log.error("Cache unavailable at %s (%s); continuing without it", self._path, exc)
            self._conn = None

    # -- setup -------------------------------------------------------------- #

    def _configure(self) -> None:
        assert self._conn is not None
        with self._lock:
            # WAL lets readers continue while a write is in progress, and works
            # across gunicorn worker processes. busy_timeout avoids spurious
            # "database is locked" errors under concurrent writes.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.commit()

    def _create_schema(self) -> None:
        assert self._conn is not None
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version == SCHEMA_VERSION:
                return
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()

    @property
    def available(self) -> bool:
        return self._conn is not None

    # -- transcripts -------------------------------------------------------- #

    def get_transcript(self, video_id: str) -> CachedTranscript | None:
        row = self._query(
            "SELECT text, lang_code, source, fetched_at FROM transcripts WHERE video_id = ?",
            (video_id,),
        )
        if row is None:
            return None
        if self._expired(row["fetched_at"], self._transcript_ttl):
            log.debug("Transcript cache expired for %s", video_id)
            return None
        return CachedTranscript(text=row["text"], lang_code=row["lang_code"], source=row["source"])

    def put_transcript(
        self, video_id: str, *, text: str, lang_code: str | None, source: str
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO transcripts (video_id, text, lang_code, source, fetched_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (video_id, text, lang_code, source, self._clock()),
        )

    # -- summaries ---------------------------------------------------------- #

    @staticmethod
    def summary_key(
        video_id: str,
        *,
        model: str,
        prompt_version: int,
        max_chars: int,
        lang_code: str | None = None,
    ) -> str:
        """Key the summary on everything that can change its text.

        Without the model and prompt version here, editing the prompt would
        silently keep serving summaries produced by the old one.
        """
        raw = f"{video_id}|{model}|{prompt_version}|{max_chars}|{lang_code or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_summary(self, cache_key: str) -> CachedSummary | None:
        row = self._query(
            "SELECT summary, truncated, transcript_chars, created_at"
            " FROM summaries WHERE cache_key = ?",
            (cache_key,),
        )
        if row is None:
            return None
        if self._expired(row["created_at"], self._summary_ttl):
            return None
        return CachedSummary(
            summary=row["summary"],
            truncated=bool(row["truncated"]),
            transcript_chars=row["transcript_chars"],
        )

    def put_summary(
        self,
        cache_key: str,
        *,
        video_id: str,
        summary: str,
        truncated: bool,
        transcript_chars: int,
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO summaries"
            " (cache_key, video_id, summary, truncated, transcript_chars, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (cache_key, video_id, summary, int(truncated), transcript_chars, self._clock()),
        )

    # -- failures ----------------------------------------------------------- #

    def get_failure(self, video_id: str) -> CachedFailure | None:
        row = self._query(
            "SELECT status, code, message, created_at FROM failures WHERE video_id = ?",
            (video_id,),
        )
        if row is None:
            return None
        if self._expired(row["created_at"], self._failure_ttl):
            return None
        return CachedFailure(status=row["status"], code=row["code"], message=row["message"])

    def put_failure(
        self, video_id: str, *, status: int, code: str, message: str
    ) -> None:
        """Remember a *transcript-stage* failure briefly.

        Cached failures exist to stop us hammering a provider that is blocking
        us. They are deliberately short-lived: a transient block must not be
        baked in for days.
        """
        self._write(
            "INSERT OR REPLACE INTO failures (video_id, status, code, message, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (video_id, status, code, message, self._clock()),
        )

    def clear_failure(self, video_id: str) -> None:
        self._write("DELETE FROM failures WHERE video_id = ?", (video_id,))

    # -- maintenance -------------------------------------------------------- #

    def purge_expired(self) -> int:
        if self._conn is None:
            return 0
        now = self._clock()
        removed = 0
        with self._lock:
            try:
                removed += self._conn.execute(
                    "DELETE FROM transcripts WHERE fetched_at < ?",
                    (now - self._transcript_ttl,),
                ).rowcount
                removed += self._conn.execute(
                    "DELETE FROM summaries WHERE created_at < ?",
                    (now - self._summary_ttl,),
                ).rowcount
                removed += self._conn.execute(
                    "DELETE FROM failures WHERE created_at < ?",
                    (now - self._failure_ttl,),
                ).rowcount
                self._conn.commit()
            except sqlite3.Error as exc:
                log.warning("Cache purge failed: %s", exc)
                return 0
        if removed:
            log.info("Purged %d expired cache rows", removed)
        return removed

    def stats(self) -> dict[str, int]:
        if self._conn is None:
            return {"enabled": 0, "transcripts": 0, "summaries": 0, "failures": 0}
        with self._lock:
            try:
                counts = {
                    table: self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
                    for table in ("transcripts", "summaries", "failures")
                }
            except sqlite3.Error:
                return {"enabled": 1, "transcripts": 0, "summaries": 0, "failures": 0}
        return {"enabled": 1, **counts}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- internals ---------------------------------------------------------- #

    def _expired(self, stored_at: float, ttl: int) -> bool:
        return ttl > 0 and (self._clock() - stored_at) > ttl

    def _query(self, sql: str, params: tuple) -> sqlite3.Row | None:
        if self._conn is None:
            return None
        with self._lock:
            try:
                return self._conn.execute(sql, params).fetchone()
            except sqlite3.Error as exc:
                log.warning("Cache read failed: %s", exc)
                return None

    def _write(self, sql: str, params: tuple) -> None:
        if self._conn is None:
            return

        should_purge = False
        with self._lock:
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
            except sqlite3.Error as exc:
                log.warning("Cache write failed: %s", exc)
                return

            self._writes += 1
            # Opportunistic housekeeping so stale rows do not accumulate forever.
            should_purge = self._writes % 200 == 0

        if should_purge:
            self.purge_expired()


def build_cache(
    path: str | Path,
    *,
    enabled: bool = True,
    transcript_ttl: int = 90 * 24 * 3600,
    summary_ttl: int = 90 * 24 * 3600,
    failure_ttl: int = 600,
    clock: Callable[[], float] = time.time,
) -> SummaryCache | NullCache:
    """Return a real cache, or a no-op one when caching is disabled."""
    if not enabled:
        log.info("Cache disabled; every request will refetch and re-summarize")
        return NullCache()
    return SummaryCache(
        path,
        transcript_ttl=transcript_ttl,
        summary_ttl=summary_ttl,
        failure_ttl=failure_ttl,
        clock=clock,
    )
