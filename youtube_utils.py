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
    try:
        print(f"DEBUG: Downloading subtitle from: {url}")
        response = requests.get(url, timeout=10)
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

    except Exception as e:
        print(f"DEBUG: Error downloading/parsing subtitle: {e}")
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


def extract_transcript(youtube_url, timeout_seconds=30):
    """
    Extracts YouTube video transcript using yt-dlp as primary method,
    with YouTube Data API as fallback.
    It will attempt to fetch English, then Russian, then French transcripts.
    """
    print(f"DEBUG: extract_transcript called for URL: {youtube_url}")

    if youtube_url in transcript_cache:
        print(f"DEBUG: Returning cached transcript for {youtube_url}")
        return transcript_cache[youtube_url]

    # Define the order of preferred languages
    preferred_languages = ['en', 'ru', 'fr']

    def _extract_with_timeout():
        """Inner function to handle transcript extraction with timeout."""
        # First attempt: yt-dlp method
        print("DEBUG: Attempting transcript extraction via yt-dlp...")
        for lang in preferred_languages:
            print(
                f"DEBUG: Attempting to fetch transcript in '{lang}' using yt-dlp...")
            try:
                # Try multiple yt-dlp configurations to extract transcripts
                # First attempt: Direct subtitle extraction
                ydl_opts = {
                    'skip_download': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,  # Include auto-generated subtitles
                    # Try target lang + English as fallback
                    'subtitleslangs': [lang, 'en'],
                    'subtitlesformat': 'vtt',  # Use vtt format which is easier to parse
                    'quiet': True,
                    'no_warnings': True,
                    'noplaylist': True,
                    'ignoreerrors': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(youtube_url, download=False)
                    print(
                        f"DEBUG: yt-dlp result keys: {list(result.keys()) if result else 'No result'}")

                    # Method 1: Check if subtitles are directly available in the result
                    transcript_text = None

                    # Check manual subtitles first
                    if 'subtitles' in result and result['subtitles']:
                        print(
                            f"DEBUG: Manual subtitles available: {list(result['subtitles'].keys())}")
                        # Try preferred language first, then English
                        for try_lang in [lang, 'en']:
                            if try_lang in result['subtitles']:
                                subtitles = result['subtitles'][try_lang]
                                if subtitles:
                                    # Get the first available subtitle format
                                    for subtitle in subtitles:
                                        if subtitle.get('ext') == 'vtt':
                                            transcript_text = download_and_parse_subtitle(
                                                subtitle['url'])
                                            if transcript_text:
                                                print(
                                                    f"DEBUG: Successfully extracted {try_lang} manual transcript")
                                                transcript_cache[youtube_url] = (
                                                    transcript_text, try_lang)
                                                return transcript_text, try_lang

                    # Method 2: Check automatic captions
                    if not transcript_text and 'automatic_captions' in result and result['automatic_captions']:
                        print(
                            f"DEBUG: Automatic captions available: {list(result['automatic_captions'].keys())}")
                        for try_lang in [lang, 'en']:
                            if try_lang in result['automatic_captions']:
                                captions = result['automatic_captions'][try_lang]
                                if captions:
                                    for caption in captions:
                                        if caption.get('ext') == 'vtt':
                                            transcript_text = download_and_parse_subtitle(
                                                caption['url'])
                                            if transcript_text:
                                                print(
                                                    f"DEBUG: Successfully extracted {try_lang} auto-generated transcript")
                                                transcript_cache[youtube_url] = (
                                                    transcript_text, try_lang)
                                                return transcript_text, try_lang

                    # Method 3: Try to extract from description or other metadata as last resort
                    if not transcript_text and 'description' in result and result['description']:
                        # Use video description as fallback (very basic, but better than nothing)
                        description = result['description']
                        if len(description) > 100:  # Only use if it's substantial
                            print(
                                f"DEBUG: Using video description as fallback transcript")
                            transcript_cache[youtube_url] = (description, 'en')
                            return description, 'en'

                    print(
                        f"DEBUG: No transcript found for {lang} using yt-dlp")

            except yt_dlp.utils.DownloadError as e:
                # This error is often verbose and can indicate that no transcript is available in the requested language.
                print(f"DEBUG: yt-dlp download error for lang '{lang}': {e}")
                # Continue to the next language
                continue
            except Exception as e:
                print(
                    f"DEBUG: An unexpected error occurred for lang '{lang}': {type(e).__name__}: {e}")
                # Continue to the next language
                continue

        # Second attempt: YouTube Data API method (if configured)
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
        return None, "No suitable transcript found for this video in English, Russian, or French."

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
