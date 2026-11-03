"""YouTube transcript retrieval.

Two strategies, tried in order:
  1. ``youtube-transcript-api`` (fast, no page scraping)
  2. ``yt-dlp`` subtitle extraction (handles more videos, needs no API key)

Deliberate choices compared to the original implementation:
  * No global TLS-verification bypass and no ``nocheckcertificate``.
  * Every outbound call has an explicit timeout and a response size cap.
  * The parsers are pure functions, so they are unit-testable without network.
  * Failures raise typed exceptions instead of overloading a return tuple.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as yta_errors

log = logging.getLogger(__name__)

# Error types are named differently across youtube-transcript-api majors (0.x vs
# 1.x) and new ones appear as YouTube changes. Resolve them defensively so an
# upgrade cannot silently turn an *expected* failure into an "unexpected" one.
_EXPECTED_API_ERRORS: tuple[type[BaseException], ...] = tuple(
    candidate
    for candidate in (
        getattr(yta_errors, name, None)
        for name in (
            "CouldNotRetrieveTranscript",  # 1.x base class
            "TranscriptsDisabled",
            "NoTranscriptFound",
            "VideoUnavailable",
            "VideoUnplayable",
            "AgeRestricted",
            "IpBlocked",
            "RequestBlocked",
            "PoTokenRequired",
            "YouTubeRequestFailed",
            "YouTubeDataUnparsable",
        )
    )
    if isinstance(candidate, type)
)

# "This language is not available" is a normal control-flow outcome, not an error:
# it means "try the next strategy", so it must never surface as a failure.
_NO_TRANSCRIPT_FOUND: tuple[type[BaseException], ...] = tuple(
    candidate
    for candidate in (
        getattr(yta_errors, "NoTranscriptFound", None),
        getattr(yta_errors, "CouldNotRetrieveTranscript", None),
    )
    if isinstance(candidate, type)
)

DEFAULT_PREFERRED_LANGUAGES: tuple[str, ...] = ("en", "ru", "fr")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Subtitle files are text; anything beyond this is a sign of something wrong.
MAX_SUBTITLE_BYTES = 5_000_000

_VIDEO_ID_RE = re.compile(
    r"""(?:https?://)?                       # optional scheme
        (?:www\.|m\.|music\.)?                # optional www/mobile subdomain
        (?:
            youtube\.com/(?:                  # /watch?v=, /v/, /embed/, /shorts/, /live/
                [^/]+/\S+?/video
              | (?:v|e(?:mbed)?|shorts|live)/
              | \S*?[?&]v=
            )
          | youtu\.be/
        )
        (?P<id>[A-Za-z0-9_-]{11})
        (?:[?&#]|$)""",
    re.VERBOSE,
)

_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TranscriptError(RuntimeError):
    """Base class for transcript problems that we can explain to the user."""


class InvalidVideoUrl(TranscriptError):
    pass


class NoTranscriptAvailable(TranscriptError):
    pass


class TranscriptFetchError(TranscriptError):
    pass


@dataclass(frozen=True)
class Transcript:
    text: str
    lang_code: str | None = None
    source: str = "unknown"  # "youtube-api" | "yt-dlp"


# --------------------------------------------------------------------------- #
# Video id
# --------------------------------------------------------------------------- #


def extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube id, or None if the URL is not a video.

    Accepts watch URLs, youtu.be short links, /shorts/, /embed/, /live/, and a
    bare id.
    """
    if not isinstance(url, str):
        return None
    candidate = url.strip()
    if not candidate:
        return None
    if _BARE_ID_RE.match(candidate):
        return candidate

    match = _VIDEO_ID_RE.search(candidate)
    return match.group("id") if match else None


# --------------------------------------------------------------------------- #
# Pure parsers
# --------------------------------------------------------------------------- #

_TIME_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def parse_json3(payload: dict[str, Any]) -> str:
    """Parse YouTube's ``json3`` caption format.

    Each event is one caption *line* and carries no trailing space, so events
    must be joined with a separator — concatenating the raw segments runs the
    last word of one line into the first word of the next ("elephantsthe").
    Segments *within* an event are word fragments and are joined as-is.
    """
    lines: list[str] = []
    for event in payload.get("events") or ():
        if not isinstance(event, dict):
            continue
        segments = event.get("segs")
        if not segments:
            continue
        piece = "".join(
            segment.get("utf8", "")
            for segment in segments
            if isinstance(segment, dict) and segment.get("utf8")
        )
        if piece.strip():
            lines.append(piece)
    return _normalize(" ".join(lines))


_WEBVTT_MARKER = "WEBVTT"


def _parse_cues(raw: str) -> str:
    """Shared cue-body parser for VTT and SRT."""
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("WEBVTT", "NOTE", "Kind:", "Language:", "STYLE", "REGION")):
            continue
        if "-->" in stripped:
            continue
        if stripped.isdigit():  # bare cue index (SRT)
            continue

        cleaned = _normalize(_TIME_TAG_RE.sub("", stripped))
        if cleaned and (not lines or lines[-1] != cleaned):
            lines.append(cleaned)

    return _normalize(" ".join(lines))


def parse_vtt(raw: str) -> str:
    """Parse WebVTT captions into plain text, or return "" if it isn't VTT.

    The header check matters: YouTube sometimes serves an HLS playlist
    (``#EXTM3U``) from a track advertised as ``vtt``. Without this guard the
    playlist body would be accepted as the transcript.
    """
    if not raw:
        return ""

    head = raw[:512].lstrip("\ufeff \t\r\n")
    if not head.startswith(_WEBVTT_MARKER):
        return ""

    return _parse_cues(raw)


def parse_srt(raw: str) -> str:
    """Parse SubRip captions into plain text, or return "" if it isn't SRT."""
    if not raw or "-->" not in raw:
        return ""
    return _parse_cues(raw)


# --------------------------------------------------------------------------- #
# Subtitle fetching
# --------------------------------------------------------------------------- #


def _http_get(url: str, timeout: float, max_bytes: int = MAX_SUBTITLE_BYTES) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TranscriptFetchError(f"Could not download subtitles: {exc}") from exc

    if len(data) > max_bytes:
        raise TranscriptFetchError(
            f"Subtitle file exceeded the {max_bytes // 1_000_000} MB limit."
        )
    return data


def _parse_subtitle_tracks(tracks: Iterable[dict[str, Any]], url: str, timeout: float) -> str | None:
    """Try every track in preference order: json3, then vtt, then srt."""
    tracks = [t for t in tracks if isinstance(t, dict)]

    for track in tracks:
        if track.get("ext") != "json3" or not track.get("url"):
            continue
        raw = _http_get(track["url"], timeout)
        try:
            text = parse_json3(json.loads(raw.decode("utf-8", errors="replace")))
        except json.JSONDecodeError:
            log.debug("Malformed json3 subtitles for %s", url)
            continue
        if text:
            return text

    for extension, parser in (("vtt", parse_vtt), ("srt", parse_srt)):
        for track in tracks:
            if track.get("ext") != extension or not track.get("url"):
                continue
            raw = _http_get(track["url"], timeout)
            text = parser(raw.decode("utf-8", errors="replace"))
            if text:
                return text
            log.debug(
                "Track for %s claimed %s but did not parse as %s; skipping",
                url,
                extension,
                extension,
            )

    return None


def _pick_tracks(
    available: dict[str, list[dict[str, Any]]], preferred: Sequence[str]
) -> list[str]:
    """Order language codes: preferred first, then everything else."""
    ordered: list[str] = [lang for lang in preferred if lang in available]
    ordered.extend(lang for lang in available if lang not in ordered)
    return ordered


# yt-dlp reports a permanent content problem with these phrases; everything else
# is treated as a possible extractor/anti-bot failure where upgrading helps.
_UNAVAILABLE_HINTS = (
    "video unavailable",
    "video is unavailable",
    "private video",
    "has been removed",
    "does not exist",
    "members-only",
    "not available in your country",
    "blocked it in your country",
    "removed by the uploader",
    "account associated with this video has been terminated",
)


# Connectivity problems are neither "video is gone" nor "yt-dlp is stale", and
# blaming the version for a dead proxy sends people down the wrong path.
_NETWORK_HINTS = (
    "unable to connect to proxy",
    "failed to establish a new connection",
    "connection refused",
    "connection reset",
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "getaddrinfo failed",
    "timed out",
)


def _looks_unavailable(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _UNAVAILABLE_HINTS)


def _looks_like_a_network_problem(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _NETWORK_HINTS)


# --------------------------------------------------------------------------- #
# Strategy 1: youtube-transcript-api
# --------------------------------------------------------------------------- #


def _fetch_via_api(video_id: str, preferred: Sequence[str]) -> Transcript:
    """Use youtube-transcript-api, tolerating both the 0.x and 1.x interfaces."""
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):  # 0.x
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    else:  # 1.x
        transcript_list = YouTubeTranscriptApi().list(video_id)

    # "That language is not available" is routine: fall through to the next option.
    transcript = None
    with contextlib.suppress(*_NO_TRANSCRIPT_FOUND):
        transcript = transcript_list.find_manually_created_transcript(list(preferred))

    if transcript is None:
        with contextlib.suppress(*_NO_TRANSCRIPT_FOUND):
            transcript = transcript_list.find_generated_transcript(list(preferred))

    if transcript is None:
        try:
            transcript = next(iter(transcript_list))
        except StopIteration:
            raise NoTranscriptAvailable("This video has no transcripts at all.") from None

    fetched = transcript.fetch()
    pieces: list[str] = []
    for entry in fetched:
        # 0.x yields dicts, 1.x yields objects with .text.
        text = getattr(entry, "text", None)
        if text is None and isinstance(entry, dict):
            text = entry.get("text")
        if text:
            pieces.append(str(text))

    body = _normalize(" ".join(pieces))
    if not body:
        raise NoTranscriptAvailable("The transcript for this video is empty.")

    return Transcript(text=body, lang_code=transcript.language_code, source="youtube-api")


# --------------------------------------------------------------------------- #
# Strategy 2: yt-dlp
# --------------------------------------------------------------------------- #


def _fetch_via_ytdlp(
    video_url: str,
    preferred: Sequence[str],
    timeout: float,
    cookie_file: Path | None,
) -> Transcript:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": timeout,
        "user_agent": USER_AGENT,
        "retries": 2,
        "noplaylist": True,
        # Keep yt-dlp from writing to a shared cache directory.
        "cachedir": False,
    }
    if cookie_file is not None and cookie_file.is_file():
        options["cookiefile"] = str(cookie_file)
        log.debug("Using cookie file %s", cookie_file)

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        # Distinguish "this video is gone" (nothing will ever help) from
        # "YouTube changed and yt-dlp is stale" (upgrading fixes it).
        message = str(exc)

        if _looks_unavailable(message):
            log.info("Video unavailable via yt-dlp: %s", exc)
            raise NoTranscriptAvailable(
                "This video is unavailable, private, deleted, or blocked in this "
                "region. Check the URL in a browser."
            ) from exc

        if _looks_like_a_network_problem(message):
            log.warning("Network problem reaching YouTube: %s", exc)
            raise TranscriptFetchError(
                "Could not reach YouTube from this server. Check the network or "
                f"proxy configuration ({exc})"
            ) from exc

        # YouTube's anti-bot responses ("The page needs to be reloaded",
        # "Failed to extract any player response") almost always mean yt-dlp is
        # out of date, which is otherwise very hard to diagnose from the message.
        raise TranscriptFetchError(
            f"yt-dlp {yt_dlp.version.__version__} could not read this video: {exc} "
            "This usually means yt-dlp is out of date; try: pip install -U yt-dlp"
        ) from exc

    if not isinstance(info, dict):
        raise TranscriptFetchError("yt-dlp returned no video information.")

    manual = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    for pool in (manual, automatic):
        if not isinstance(pool, dict):
            continue
        for lang in _pick_tracks(pool, preferred):
            text = _parse_subtitle_tracks(pool[lang], video_url, timeout)
            if text:
                source = "yt-dlp" if pool is manual else "yt-dlp (auto)"
                return Transcript(text=text, lang_code=lang, source=source)

    raise NoTranscriptAvailable("This video has no usable subtitles.")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def fetch_transcript(
    video_url: str,
    *,
    http_timeout: float = 20.0,
    preferred_languages: Sequence[str] = DEFAULT_PREFERRED_LANGUAGES,
    cookie_file: Path | None = None,
) -> Transcript:
    """Resolve a YouTube URL to plain-text captions, or raise TranscriptError."""
    video_id = extract_video_id(video_url)
    if video_id is None:
        raise InvalidVideoUrl(
            "That does not look like a YouTube video URL. "
            "Expected something like https://www.youtube.com/watch?v=XXXXXXXXXXX"
        )

    preferred = list(preferred_languages) or list(DEFAULT_PREFERRED_LANGUAGES)

    try:
        transcript = _fetch_via_api(video_id, preferred)
        log.info(
            "Transcript for %s via youtube-transcript-api (%s, %d chars)",
            video_id,
            transcript.lang_code,
            len(transcript.text),
        )
        return transcript
    except _EXPECTED_API_ERRORS as exc:
        # Expected: disabled captions, no matching language, IP block, ...
        log.info(
            "youtube-transcript-api unusable for %s (%s: %s); trying yt-dlp",
            video_id,
            type(exc).__name__,
            exc,
        )
    except NoTranscriptAvailable as exc:
        log.info("No transcript via API for %s (%s); trying yt-dlp", video_id, exc)
    except TranscriptFetchError as exc:
        log.info("Transcript fetch error for %s (%s); trying yt-dlp", video_id, exc)
    except Exception as exc:  # noqa: BLE001 - third-party library is inconsistent
        log.warning("Unexpected youtube-transcript-api failure for %s: %s", video_id, exc)

    transcript = _fetch_via_ytdlp(video_url, preferred, http_timeout, cookie_file)
    log.info(
        "Transcript for %s via yt-dlp (%s, %d chars)",
        video_id,
        transcript.lang_code,
        len(transcript.text),
    )
    return transcript
