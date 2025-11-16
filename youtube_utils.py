import yt_dlp
import json
import os
import time
import signal
import threading
import re
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Transcript extraction timed out")


transcript_cache = {}
language_cache = {}  # Cache for language codes separately
comments_cache = {}
youtube = None


def configure_youtube_api():
    """Initializes the YouTube API client."""
    global youtube
    api_key = os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in environment variables.")
    youtube = build('youtube', 'v3', developerKey=api_key)


def _call_youtube_api_with_retries(api_call, max_retries=5, backoff_factor=1.5):
    """
    Helper function to call YouTube API with retry and exponential backoff.
    """
    for i in range(max_retries):
        try:
            return api_call.execute()
        except HttpError as e:
            if e.resp.status == 429:
                sleep_time = backoff_factor ** i
                print(
                    f"Rate limit hit. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                raise
    raise Exception(f"YouTube API call failed after {max_retries} retries.")


def extract_transcript_youtube_api(youtube_url):
    """
    Extract transcript using YouTube Data API as primary method.
    Returns tuple of (transcript_text, lang_code) or (None, error_message)
    """
    print(
        f"DEBUG: Attempting YouTube API transcript extraction for: {youtube_url}")

    # Extract video ID from URL
    video_id_match = re.search(
        r'(?:v=|/|watch\?v=)([a-zA-Z0-9_-]{11})', youtube_url)
    if not video_id_match:
        return None, "Invalid YouTube URL format"

    video_id = video_id_match.group(1)
    print(f"DEBUG: Extracted video ID: {video_id}")

    if not youtube:
        return None, "YouTube API client not configured"

    try:
        # Get available caption tracks
        captions_request = youtube.captions().list(
            part="snippet",
            videoId=video_id
        )
        captions_response = _call_youtube_api_with_retries(captions_request)

        if not captions_response.get('items'):
            return None, "No captions available for this video via YouTube API"

        print(f"DEBUG: Found {len(captions_response['items'])} caption tracks")

        # Try preferred languages in order
        target_langs = ['en', 'ru', 'fr']
        transcript_text = None
        lang_code = None

        for lang in target_langs:
            # Find caption track for this language
            caption_track = None
            for item in captions_response['items']:
                if item['snippet']['language'] == lang:
                    caption_track = item
                    break

            if caption_track:
                print(f"DEBUG: Found {lang} caption track, downloading...")
                try:
                    # Download caption content
                    download_request = youtube.captions().download(
                        id=caption_track['id'],
                        tfmt='srt'  # Get in SRT format for easier parsing
                    )
                    caption_content = _call_youtube_api_with_retries(
                        download_request)

                    if caption_content:
                        # Parse SRT format
                        transcript_text = parse_srt_captions(caption_content)
                        lang_code = lang
                        print(
                            f"DEBUG: Successfully extracted {lang} transcript via YouTube API")
                        break
                except Exception as e:
                    print(f"DEBUG: Failed to download {lang} captions: {e}")
                    continue

        if transcript_text:
            return transcript_text, lang_code
        else:
            return None, "No suitable captions found via YouTube API"

    except Exception as e:
        print(f"DEBUG: YouTube API transcript extraction failed: {e}")
        return None, f"YouTube API error: {e}"


def parse_srt_captions(srt_content):
    """
    Parse SRT format captions and extract text content.
    """
    try:
        # Simple SRT parsing - remove timestamps and sequence numbers
        lines = srt_content.split('\n')
        text_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines, sequence numbers, and timestamp lines
            if not line or line.isdigit() or '-->' in line:
                continue
            text_lines.append(line)

        return ' '.join(text_lines)
    except Exception as e:
        print(f"DEBUG: Error parsing SRT captions: {e}")
        return None


def parse_vtt_captions(vtt_content):
    """
    Parse VTT format captions and extract text content.
    """
    try:
        lines = vtt_content.split('\n')
        text_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines, timestamp lines, and VTT header
            if not line or '-->' in line or line == 'WEBVTT':
                continue
            text_lines.append(line)

        return ' '.join(text_lines)
    except Exception as e:
        print(f"DEBUG: Error parsing VTT captions: {e}")
        return None


def download_and_parse_subtitle(url):
    """
    Download subtitle from URL and parse it.
    """
    max_retries = 3
    backoff_factor = 2

    for attempt in range(max_retries):
        try:
            print(
                f"DEBUG: Downloading subtitle from: {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            if url.endswith('.vtt'):
                return parse_vtt_captions(response.text)
            elif url.endswith('.srt'):
                return parse_srt_captions(response.text)
            else:
                # Try to detect format
                if 'WEBVTT' in response.text:
                    return parse_vtt_captions(response.text)
                else:
                    return parse_srt_captions(response.text)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(
                        f"DEBUG: Rate limit hit, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(
                        f"DEBUG: Max retries exceeded for rate limiting: {e}")
                    return None
            else:
                print(f"DEBUG: HTTP error downloading subtitle: {e}")
                return None
        except Exception as e:
            print(f"DEBUG: Error downloading/parsing subtitle: {e}")
            return None

    return None


def get_comments(video_id):
    """Fetch comments from YouTube Data API, with caching.
    Returns tuple of (comments_list, error_message) or (None, error_message) on error.
    """
    if video_id in comments_cache:
        print(f"Returning cached comments for video ID: {video_id}")
        return comments_cache[video_id], None

    comments = []
    print("Fetching comments. This can exhaust API quotas quickly on popular videos.")

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        )
        results = _call_youtube_api_with_retries(request)

        while results:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            if 'nextPageToken' in results:
                print(
                    "Note: Only fetching the first page of comments to conserve API quota.")
                break
            else:
                break

        if not comments:
            return None, "No comments found for this video or comments are disabled."

    except Exception as e:
        error_msg = f"Error fetching comments: {str(e)}"
        print(error_msg)
        return None, error_msg

    print(f"Caching {len(comments)} comments for video ID: {video_id}")
    comments_cache[video_id] = comments
    return comments, None


def detect_video_language(available_captions):
    """
    Analyzes available captions to determine the video's primary language.
    Returns the most likely language code.
    """
    if not available_captions:
        return 'en'  # Default to English

    # Look for language patterns that indicate primary language
    caption_languages = list(available_captions.keys())
    print(f"DEBUG: Available caption languages: {caption_languages}")

    # Check for original/origin language indicators
    for lang in caption_languages:
        if '-orig' in lang:
            base_lang = lang.split('-')[0]
            print(
                f"DEBUG: Found original language indicator: {lang} -> {base_lang}")
            return base_lang

    # Check for common language patterns
    # If we have both 'en' and 'ru', prefer the one that appears first in YouTube's ordering
    # YouTube typically lists the primary language first in automatic captions

    # Count occurrences to find the most common language
    lang_counts = {}
    for lang in caption_languages:
        base_lang = lang.split('-')[0]  # Handle 'en-orig', 'ru', etc.
        lang_counts[base_lang] = lang_counts.get(base_lang, 0) + 1

    # If we have clear primary language indicators, use them
    if 'en' in lang_counts and 'ru' in lang_counts:
        # If English appears before Russian in the list, it's likely an English video
        en_index = caption_languages.index(
            'en') if 'en' in caption_languages else float('inf')
        ru_index = caption_languages.index(
            'ru') if 'ru' in caption_languages else float('inf')

        if en_index < ru_index:
            print("DEBUG: English appears before Russian, likely English video")
            return 'en'
        else:
            print("DEBUG: Russian appears before English, likely Russian video")
            return 'ru'

    # Default to most common language, or English if uncertain
    if lang_counts:
        primary_lang = max(lang_counts.items(), key=lambda x: x[1])[0]
        print(f"DEBUG: Using most common language: {primary_lang}")
        return primary_lang

    return 'en'


def extract_transcript(youtube_url, timeout_seconds=30):
    """
    Extracts YouTube video transcript using yt-dlp as primary method,
    with YouTube Data API as fallback.
    It will detect the video's primary language and extract transcripts accordingly.
    """
    print(f"DEBUG: extract_transcript called for URL: {youtube_url}")

    if youtube_url in transcript_cache:
        print(f"DEBUG: Returning cached transcript for {youtube_url}")
        return transcript_cache[youtube_url]

    def _extract_with_timeout():
        """Inner function to handle transcript extraction with timeout."""
        # First attempt: yt-dlp method
        print("DEBUG: Attempting transcript extraction via yt-dlp...")

        try:
            # First, get video info to detect language
            ydl_opts_info = {
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'ignoreerrors': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                result = ydl.extract_info(youtube_url, download=False)

                # Detect video language from available captions
                available_captions = result.get('automatic_captions', {})
                if not available_captions:
                    available_captions = result.get('subtitles', {})

                detected_language = detect_video_language(available_captions)
                print(f"DEBUG: Detected video language: {detected_language}")

                # Set preferred languages based on detection
                if detected_language == 'ru':
                    # Russian video, try Russian first
                    preferred_languages = ['ru', 'en']
                else:
                    # Non-Russian video, try English first
                    preferred_languages = ['en', 'ru']

                print(f"DEBUG: Using language priority: {preferred_languages}")

                for lang in preferred_languages:
                    print(
                        f"DEBUG: Attempting to fetch transcript in '{lang}' using yt-dlp...")

                    ydl_opts = {
                        'skip_download': True,
                        'writesubtitles': True,
                        'writeautomaticsub': True,
                        'subtitleslangs': [lang],
                        'subtitlesformat': 'vtt',
                        'quiet': True,
                        'no_warnings': True,
                        'noplaylist': True,
                        'ignoreerrors': True,
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl_lang:
                        result_lang = ydl_lang.extract_info(
                            youtube_url, download=False)

                        transcript_text = None

                        # Check manual subtitles first
                        if 'subtitles' in result_lang and result_lang['subtitles']:
                            if lang in result_lang['subtitles']:
                                subtitles = result_lang['subtitles'][lang]
                                if subtitles:
                                    for subtitle in subtitles:
                                        if subtitle.get('ext') == 'vtt':
                                            transcript_text = download_and_parse_subtitle(
                                                subtitle['url'])
                                            if transcript_text:
                                                print(
                                                    f"DEBUG: Successfully extracted {lang} manual transcript")
                                                transcript_cache[youtube_url] = (
                                                    transcript_text, lang)
                                                return transcript_text, lang

                        # Check automatic captions
                        if not transcript_text and 'automatic_captions' in result_lang and result_lang['automatic_captions']:
                            if lang in result_lang['automatic_captions']:
                                captions = result_lang['automatic_captions'][lang]
                                if captions:
                                    for caption in captions:
                                        if caption.get('ext') == 'vtt':
                                            transcript_text = download_and_parse_subtitle(
                                                caption['url'])
                                            if transcript_text:
                                                print(
                                                    f"DEBUG: Successfully extracted {lang} auto-generated transcript")
                                                transcript_cache[youtube_url] = (
                                                    transcript_text, lang)
                                                return transcript_text, lang

                        print(
                            f"DEBUG: No transcript found for {lang} using yt-dlp")

        except Exception as e:
            print(
                f"DEBUG: Error in language detection or initial extraction: {type(e).__name__}: {e}")

        # Fallback: Try standard languages in order
        print("DEBUG: Language-based extraction failed, trying fallback order...")
        fallback_languages = ['en', 'ru', 'fr']

        for lang in fallback_languages:
            print(f"DEBUG: Fallback attempt for '{lang}'...")
            try:
                ydl_opts = {
                    'skip_download': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': [lang],
                    'subtitlesformat': 'vtt',
                    'quiet': True,
                    'no_warnings': True,
                    'noplaylist': True,
                    'ignoreerrors': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(youtube_url, download=False)

                    transcript_text = None

                    # Check manual subtitles
                    if 'subtitles' in result and result['subtitles']:
                        if lang in result['subtitles']:
                            subtitles = result['subtitles'][lang]
                            if subtitles:
                                for subtitle in subtitles:
                                    if subtitle.get('ext') == 'vtt':
                                        transcript_text = download_and_parse_subtitle(
                                            subtitle['url'])
                                        if transcript_text:
                                            print(
                                                f"DEBUG: Successfully extracted {lang} manual transcript (fallback)")
                                            transcript_cache[youtube_url] = (
                                                transcript_text, lang)
                                            return transcript_text, lang

                    # Check automatic captions
                    if not transcript_text and 'automatic_captions' in result and result['automatic_captions']:
                        if lang in result['automatic_captions']:
                            captions = result['automatic_captions'][lang]
                            if captions:
                                for caption in captions:
                                    if caption.get('ext') == 'vtt':
                                        transcript_text = download_and_parse_subtitle(
                                            caption['url'])
                                        if transcript_text:
                                            print(
                                                f"DEBUG: Successfully extracted {lang} auto-generated transcript (fallback)")
                                            transcript_cache[youtube_url] = (
                                                transcript_text, lang)
                                            return transcript_text, lang

            except Exception as e:
                print(
                    f"DEBUG: Error in fallback extraction for {lang}: {type(e).__name__}: {e}")
                continue

        # Final fallback: YouTube Data API
        print("DEBUG: yt-dlp method failed, attempting YouTube Data API fallback...")
        try:
            transcript_text, lang_code = extract_transcript_youtube_api(
                youtube_url)
            if transcript_text:
                print(
                    f"DEBUG: Successfully extracted transcript using YouTube Data API in '{lang_code}'.")
                transcript_cache[youtube_url] = (transcript_text, lang_code)
                return transcript_text, lang_code
        except Exception as e:
            print(
                f"DEBUG: YouTube Data API fallback also failed: {type(e).__name__}: {e}")

        print("DEBUG: No suitable transcript found using any method.")
        return None, "No suitable transcript found for this video."

    # Apply timeout using threading.Timer for cross-platform compatibility
    import threading
    result_container = [None]
    exception_container = [None]

    def worker():
        try:
            result_container[0] = _extract_with_timeout()
        except Exception as e:
            exception_container[0] = e

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(
            f"DEBUG: Transcript extraction timed out after {timeout_seconds} seconds")
        return None, f"Transcript extraction timed out after {timeout_seconds} seconds. Please try again with a different video or check your network connection."

    if exception_container[0]:
        print(
            f"DEBUG: Exception during transcript extraction: {exception_container[0]}")
        return None, f"An error occurred during transcript extraction: {str(exception_container[0])}"

    return result_container[0]


def get_language_code(youtube_url):
    """Get language code from cache without re-extracting transcript."""
    if youtube_url in language_cache:
        print(f"Returning cached language code for {youtube_url}")
        return language_cache[youtube_url]

    # If not in cache, extract transcript and cache the language code
    transcript_data, lang_code = extract_transcript(youtube_url)
    if transcript_data and lang_code:
        language_cache[youtube_url] = lang_code
        return lang_code
    return None
