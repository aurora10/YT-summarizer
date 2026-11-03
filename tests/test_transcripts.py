"""Tests for video-id parsing, subtitle parsers and truncation."""

from __future__ import annotations

import pytest

from app import _prepare_transcript, estimate_tokens
from transcripts import extract_video_id, parse_json3, parse_srt, parse_vtt

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}",
        f"http://www.youtube.com/watch?v={VIDEO_ID}&t=42s",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?si=abc123",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
        VIDEO_ID,
    ],
)
def test_extract_video_id_accepts_common_forms(url: str) -> None:
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        "https://vimeo.com/123456789",
        "https://www.youtube.com/playlist?list=PLabcdefghijk",
        "https://www.youtube.com/watch?v=tooshort",
        f"https://www.youtube.com/watch?v={VIDEO_ID}extra",
        "https://www.youtube.com/channel/UCabcdefghijklmnop",
    ],
)
def test_extract_video_id_rejects_non_videos(url: str) -> None:
    assert extract_video_id(url) is None


def test_extract_video_id_handles_non_string() -> None:
    assert extract_video_id(None) is None  # type: ignore[arg-type]
    assert extract_video_id(12345) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# json3
# --------------------------------------------------------------------------- #


def test_parse_json3_separates_events() -> None:
    payload = {
        "events": [
            {"segs": [{"utf8": "Hello "}, {"utf8": "world"}]},
            {"segs": [{"utf8": "second"}]},
            {"tStartMs": 0},  # no segs -> skipped
        ]
    }
    assert parse_json3(payload) == "Hello world second"


def test_parse_json3_does_not_glue_adjacent_event_lines() -> None:
    """Real YouTube data: each event is a line with no trailing space."""
    payload = {
        "events": [
            {"segs": [{"utf8": "All right, so here we are, in front of the\nelephants"}]},
            {"segs": [{"utf8": "the cool thing about these guys"}]},
        ]
    }
    assert parse_json3(payload) == (
        "All right, so here we are, in front of the elephants "
        "the cool thing about these guys"
    )


def test_parse_json3_keeps_word_fragments_within_one_event() -> None:
    """Segments inside a single event are fragments of the same line."""
    payload = {"events": [{"segs": [{"utf8": "info"}, {"utf8": "rmation"}]}]}
    assert parse_json3(payload) == "information"


def test_parse_json3_handles_missing_events() -> None:
    assert parse_json3({}) == ""
    assert parse_json3({"events": None}) == ""


def test_parse_json3_collapses_newlines_and_spaces() -> None:
    payload = {"events": [{"segs": [{"utf8": "line one\n"}, {"utf8": "  line   two"}]}]}
    assert parse_json3(payload) == "line one line two"


# --------------------------------------------------------------------------- #
# VTT
# --------------------------------------------------------------------------- #


SAMPLE_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
Hello <b>world</b> &amp;

00:00:02.000 --> 00:00:04.000
Hello <b>world</b> &amp;

NOTE this is a comment

00:00:04.000 --> 00:00:06.000 align:start position:0%
<v Speaker>Second line</v>
"""


def test_parse_vtt_strips_markup_and_headers() -> None:
    result = parse_vtt(SAMPLE_VTT)
    assert "WEBVTT" not in result
    assert "-->" not in result
    assert "Kind:" not in result
    assert "NOTE" not in result
    assert "<b>" not in result
    assert "Second line" in result
    assert "Hello world &amp;" in result


def test_parse_vtt_dedupes_consecutive_repeated_cues() -> None:
    # The repeated "Hello world &amp;" cue should appear once.
    assert parse_vtt(SAMPLE_VTT).count("Hello world") == 1


def test_parse_vtt_ignores_bare_cue_indices() -> None:
    result = parse_vtt("WEBVTT\n\n1\n00:00.000 --> 00:01.000\ntext\n")
    assert result == "text"


def test_parse_vtt_empty() -> None:
    assert parse_vtt("") == ""
    assert parse_vtt("WEBVTT\n\n") == ""


def test_parse_vtt_rejects_hls_playlist() -> None:
    """YouTube sometimes serves an HLS playlist from a track marked as vtt.

    Accepting it would return playlist URLs as the "transcript" (found live).
    """
    playlist = (
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-PLAYLIST-TYPE:VOD\n"
        "#EXT-X-TARGETDURATION:600\n#EXTINF:18,\n"
        "https://www.youtube.com/api/timedtext?caps=asr&v=dQw4w9WgXcQ&lang=en\n"
    )
    assert parse_vtt(playlist) == ""


def test_parse_vtt_rejects_non_vtt_text() -> None:
    assert parse_vtt("just some prose, no cues here") == ""
    assert parse_vtt("<html><body>blocked</body></html>") == ""


# --------------------------------------------------------------------------- #
# SRT
# --------------------------------------------------------------------------- #


def test_parse_srt_parses_cues() -> None:
    raw = (
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "Hello <i>world</i>\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:04,000\n"
        "Hello <i>world</i>\n"
        "\n"
        "3\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "Second line\n"
    )
    assert parse_srt(raw) == "Hello world Second line"


def test_parse_srt_rejects_non_srt() -> None:
    assert parse_srt("") == ""
    assert parse_srt("no cues at all") == ""


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


def test_prepare_transcript_leaves_short_text_untouched() -> None:
    text = "short transcript"
    body, truncated = _prepare_transcript(text, 1000)
    assert body == text
    assert truncated is False


def test_prepare_transcript_cuts_on_sentence_boundary() -> None:
    sentence = "This is a sentence. "
    text = sentence * 200  # 4000 chars
    body, truncated = _prepare_transcript(text, 1000)

    assert truncated is True
    assert len(body) <= 1000
    assert body.endswith(".")
    assert text.startswith(body)
    # We stopped right before the space that follows a completed sentence.
    assert text[len(body)] == " "


def test_prepare_transcript_falls_back_to_hard_cut() -> None:
    text = "x" * 5000  # no sentence boundaries at all
    body, truncated = _prepare_transcript(text, 1000)

    assert truncated is True
    assert len(body) == 1000


def test_prepare_transcript_exact_boundary() -> None:
    text = "a" * 1000
    body, truncated = _prepare_transcript(text, 1000)
    assert body == text
    assert truncated is False


# --------------------------------------------------------------------------- #
# yt-dlp failure guidance
# --------------------------------------------------------------------------- #


def test_ytdlp_failure_mentions_the_installed_version_and_the_fix(monkeypatch) -> None:
    """An outdated yt-dlp is the usual cause, so the error must say so."""
    import yt_dlp

    from transcripts import TranscriptFetchError, _fetch_via_ytdlp

    class BoomYDL:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, *args, **kwargs):
            raise yt_dlp.utils.DownloadError("The page needs to be reloaded.")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", BoomYDL)

    with pytest.raises(TranscriptFetchError) as excinfo:
        _fetch_via_ytdlp("https://youtu.be/dQw4w9WgXcQ", ["en"], 5.0, None)

    message = str(excinfo.value)
    assert yt_dlp.version.__version__ in message
    assert "pip install -U yt-dlp" in message


def test_unavailable_video_is_not_reported_as_a_stale_ytdlp(monkeypatch) -> None:
    """A deleted video must not be blamed on yt-dlp being out of date."""
    import yt_dlp

    from transcripts import NoTranscriptAvailable, _fetch_via_ytdlp

    class BoomYDL:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, *args, **kwargs):
            raise yt_dlp.utils.DownloadError("ERROR: [youtube] X: This video is unavailable")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", BoomYDL)

    with pytest.raises(NoTranscriptAvailable) as excinfo:
        _fetch_via_ytdlp("https://youtu.be/dQw4w9WgXcQ", ["en"], 5.0, None)

    message = str(excinfo.value)
    assert "unavailable" in message.lower()
    assert "pip install -U yt-dlp" not in message


def test_network_failure_is_not_blamed_on_a_stale_ytdlp(monkeypatch) -> None:
    """A dead proxy must not be reported as "your yt-dlp is old"."""
    import yt_dlp

    from transcripts import TranscriptFetchError, _fetch_via_ytdlp

    class BoomYDL:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, *args, **kwargs):
            raise yt_dlp.utils.DownloadError(
                "Unable to download API page: ('Unable to connect to proxy', "
                "NewConnectionError('Failed to establish a new connection: "
                "[Errno 61] Connection refused'))"
            )

    monkeypatch.setattr(yt_dlp, "YoutubeDL", BoomYDL)

    with pytest.raises(TranscriptFetchError) as excinfo:
        _fetch_via_ytdlp("https://youtu.be/dQw4w9WgXcQ", ["en"], 5.0, None)

    message = str(excinfo.value)
    assert "Could not reach YouTube" in message
    assert "pip install -U yt-dlp" not in message


# --------------------------------------------------------------------------- #
# Token budgets (why a character cap alone is unsafe)
# --------------------------------------------------------------------------- #


def test_estimate_tokens_discounts_non_ascii() -> None:
    ascii_text = "a" * 4_000
    cyrillic_text = "я" * 4_000

    # Same character count, but Cyrillic costs about twice the tokens.
    assert estimate_tokens(cyrillic_text) > estimate_tokens(ascii_text) * 1.5


def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("hello world " * 10)
    long = estimate_tokens("hello world " * 1000)
    assert long > short > 0


def test_token_budget_shrinks_below_the_character_cap() -> None:
    text = "я" * 100_000  # ~45k tokens, well under a 400k char cap
    body, truncated = _prepare_transcript(text, 400_000, token_budget=10_000)

    assert truncated is True
    assert len(body) < 100_000
    assert estimate_tokens(body) <= 11_000  # roughly the budget, not the cap


def test_same_char_cap_admits_less_cyrillic_than_ascii() -> None:
    """The 2x script penalty is exactly what a character cap gets wrong."""
    ascii_text = ("word " * 25_000)[:100_000]
    cyrillic_text = ("слово " * 25_000)[:100_000]

    _, ascii_truncated = _prepare_transcript(ascii_text, 400_000, 30_000)
    _, cyrillic_truncated = _prepare_transcript(cyrillic_text, 400_000, 30_000)

    assert ascii_truncated is False  # ~25k tokens: fits
    assert cyrillic_truncated is True  # ~42k tokens: does not


def test_character_cap_still_applies_with_a_large_budget() -> None:
    text = "a" * 5_000
    body, truncated = _prepare_transcript(text, 1_000, token_budget=1_000_000)

    assert truncated is True
    assert len(body) <= 1_000


def test_no_budget_means_only_the_char_cap() -> None:
    text = "я" * 5_000
    body, truncated = _prepare_transcript(text, 1_000)

    assert truncated is True
    assert len(body) <= 1_000


def test_fitting_text_is_not_truncated() -> None:
    text = "short transcript"
    body, truncated = _prepare_transcript(text, 400_000, 128_000)

    assert (body, truncated) == (text, False)


def test_tiny_budget_still_returns_something() -> None:
    body, truncated = _prepare_transcript("a" * 10_000, 400_000, token_budget=1)

    assert truncated is True
    assert body  # never empty
