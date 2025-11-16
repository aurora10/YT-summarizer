from llm_utils import (
    chat_with_llm,
    summarize_text_with_llm,
    translate_text_with_llm,
    configure_genai,
)
from youtube_utils import (
    get_comments,
    extract_transcript,
    configure_youtube_api,
    get_language_code,
)
import os
import re
import markdown
import json
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv


def is_english_text(text):
    """
    Simple heuristic to detect if text is primarily in English.
    Returns True if text appears to be English, False otherwise.
    """
    if not text:
        return False

    # Common English words and patterns
    english_indicators = [
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with',
        'for', 'as', 'was', 'on', 'are', 'this', 'be', 'by', 'have', 'from'
    ]

    text_lower = text.lower()

    # Count English indicator words
    english_count = sum(1 for word in english_indicators if word in text_lower)

    # Check for Cyrillic characters (Russian)
    cyrillic_chars = re.findall(r'[а-яА-Я]', text)
    cyrillic_ratio = len(cyrillic_chars) / max(len(text), 1)

    # If there are many Cyrillic characters, it's likely Russian
    if cyrillic_ratio > 0.1:
        return False

    # If we found a reasonable number of English words, consider it English
    if english_count >= 3:
        return True

    # Default to English if we can't determine
    return True


# Load environment variables from .env file if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Configure the Gemini API key
try:
    configure_genai()
    print("Gemini API configured successfully.")
except ValueError as e:
    print(f"Error configuring Gemini API: {e}")

# Configure the YouTube API key
try:
    configure_youtube_api()
    print("YouTube API configured successfully.")
except ValueError as e:
    print(f"Error configuring YouTube API: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes.

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Create a directory named 'Comments' if it doesn't exist
comments_dir = "Comments"
if not os.path.exists(comments_dir):
    os.makedirs(comments_dir)


@app.route("/", methods=["GET", "POST"])
# Limit POST requests more strictly
@limiter.limit("10 per minute", methods=["POST"])
def index():
    error = None
    chat_history = ""
    video_url = ""
    previous_video_url = ""
    if request.method == "POST":
        video_url = request.form.get("video_url", "").strip()
        user_message = request.form.get("user_message", "").strip()
        chat_history = request.form.get("chat_history", "")
        clear_chat = request.form.get("clear_chat")
        previous_video_url = request.form.get("previous_video_url", "")

        print("\n--- NEW REQUEST ---")
        print(
            f"FORM DATA: video_url='{video_url}', user_message='{user_message}', previous_video_url='{previous_video_url}', clear_chat='{clear_chat}'")

        transcript_text = request.form.get("transcript_text", "")

        # --- Input Handling Logic ---
        if clear_chat:
            return render_template("index.html", error=None, chat_history="", video_url="", previous_video_url="", transcript_text="")

        if user_message and user_message.lower() == "exit":
            return render_template("index.html", error=None, chat_history="", video_url="", previous_video_url="", transcript_text="")

        is_new_video = bool(video_url) and (video_url != previous_video_url)

        # --- Processing Logic ---
        lang_code = None
        response_text = None

        if is_new_video:
            print(f"DEBUG: Processing new video: {video_url}")
            chat_history = ""
            previous_video_url = video_url
            if not user_message:
                user_message = "Summarize this video"

            print(f"DEBUG: Starting transcript extraction...")
            transcript_data, error_message_or_lang_code = extract_transcript(
                video_url)
            if transcript_data is None:
                print(
                    f"DEBUG: Transcript extraction failed: {error_message_or_lang_code}")
                error = error_message_or_lang_code or f"Could not extract transcript for {video_url}."
                previous_video_url = ""
                lang_code = None
                transcript_text = ""
            else:
                transcript_text = transcript_data
                lang_code = error_message_or_lang_code
                cleaned_transcript = transcript_text
                print(
                    f"DEBUG: Transcript extraction successful, length: {len(transcript_text)}")

                print("\n" + "="*50)
                print(
                    "DEBUG: Attempting to generate summary...")
                try:
                    print(
                        f"DEBUG: Calling summarize_text_with_llm with lang_code: {lang_code}")
                    response_text = summarize_text_with_llm(
                        cleaned_transcript, "Summarize this video", lang_code)

                    # If we have Russian transcript but got English response, force translation
                    if lang_code == 'ru' and response_text and is_english_text(response_text):
                        print(
                            "DEBUG: Russian video but got English response, translating to Russian...")
                        response_text = translate_text_with_llm(
                            response_text, 'ru')

                    if not response_text:
                        print("DEBUG: Summary generation failed.")
                        error = "Failed to generate summary from the transcript."
                except Exception as e:
                    print(
                        f"DEBUG: EXCEPTION in summarize_text_with_llm: {type(e).__name__}: {str(e)}")
                    error = "Failed to generate summary from the transcript."
                print("="*50 + "\n")

        elif user_message and previous_video_url:
            print(f"Continuing chat for video: {previous_video_url}")
            print(
                f"DEBUG: Transcript text length in follow-up: {len(transcript_text) if transcript_text else 0}")
            if not transcript_text:
                error = f"Could not find transcript for context ({previous_video_url}). Please try again with a new video."
                chat_history = ""
                video_url = previous_video_url
                previous_video_url = ""
                lang_code = None
            else:
                # Use cached language code to avoid redundant transcript extraction
                lang_code = get_language_code(previous_video_url)
                if not lang_code:
                    error = "Could not determine language for the video. Please try again with a new video."
                else:
                    cleaned_transcript = transcript_text
                    response_text = chat_with_llm(
                        cleaned_transcript, user_message, chat_history, lang_code)

        elif user_message and not previous_video_url:
            print("General chat, no video context.")
            response_text = chat_with_llm("", user_message, chat_history, "en")
            video_url = ""
            previous_video_url = ""

        if response_text:
            chat_history += f'<div class="message user-message"><span class="role">You:</span> {user_message}</div>\n'
            formatted_response = markdown.markdown(response_text)
            chat_history += f'<div class="message llm-message" lang="{lang_code}"><span class="role">LLM:</span> {formatted_response}</div>\n'

        return render_template("index.html",
                               chat_history=chat_history,
                               video_url=video_url,
                               previous_video_url=previous_video_url,
                               error=error,
                               transcript_text=transcript_text)

    return render_template("index.html", error=None, chat_history="", video_url="", previous_video_url="")


@app.route("/summarize_comments", methods=["POST"])
# Comment summarization is expensive, limit strictly
@limiter.limit("5 per minute")
def summarize_comments_route():
    video_url = request.form.get("video_url", "").strip()
    chat_history = request.form.get("chat_history", "")
    previous_video_url = request.form.get("previous_video_url", "")
    error = None
    summary_text = None

    if not video_url:
        error = "Please enter a YouTube video URL."
    else:
        try:
            # Extract video ID from URL
            video_id_match = re.search(
                r'(?:v=|"|\/|watch\?v=)([a-zA-Z0-9_-]{11})', video_url)
            if not video_id_match:
                error = "Invalid YouTube URL provided."
            else:
                video_id = video_id_match.group(1)
                print(f"Attempting to fetch comments for video ID: {video_id}")
                comments, comments_error = get_comments(video_id)

                if comments_error:
                    error = comments_error
                elif not comments:
                    error = "No comments found for this video."
                else:
                    # Join comments into a single string for summarization
                    comments_text = "\n".join(comments)
                    print(f"Fetched {len(comments)} comments. Summarizing...")

                    # Determine language for comment summarization
                    lang_code = None
                    # Try to get language from previous video URL first (if it's the same video)
                    if previous_video_url and previous_video_url == video_url:
                        lang_code = get_language_code(previous_video_url)

                    # If not available, try to detect from current video
                    if not lang_code:
                        lang_code = get_language_code(video_url)

                    # Fallback to English if still not determined
                    if not lang_code:
                        lang_code = 'en'

                    print(
                        f"DEBUG: Using language code for comment summarization: {lang_code}")

                    # Language-specific comment summarization instructions
                    comment_instructions = {
                        'en': "Summarize the following YouTube comments. Focus on common themes, sentiments, and key discussion points. Provide a concise summary in bullet points if appropriate.",
                        'ru': "Обобщите следующие комментарии YouTube. Сосредоточьтесь на общих темах, настроениях и ключевых моментах обсуждения. Предоставьте краткое резюме в виде маркированного списка, если это уместно.",
                        'fr': "Résumez les commentaires YouTube suivants. Concentrez-vous sur les thèmes communs, les sentiments et les points de discussion clés. Fournissez un résumé concis sous forme de puces si approprié."
                    }

                    prompt_instruction = comment_instructions.get(
                        lang_code, comment_instructions['en'])
                    summary_text = summarize_text_with_llm(
                        comments_text, prompt_instruction, lang_code)

                    # If we have Russian language but got English response, force translation
                    if lang_code == 'ru' and summary_text and is_english_text(summary_text):
                        print(
                            "DEBUG: Russian video comments but got English summary, translating to Russian...")
                        summary_text = translate_text_with_llm(
                            summary_text, 'ru')

                    print("Comments summarization complete.")

                    if summary_text:
                        # Add the summary to the chat history with language code
                        chat_history += f'<div class="message user-message"><span class="role">You:</span> Summarize comments for {video_url}</div>\n'
                        formatted_summary = markdown.markdown(summary_text)
                        chat_history += f'<div class="message llm-message" lang="{lang_code}"><span class="role">LLM (Comments Summary):</span> {formatted_summary}</div>\n'
                    else:
                        error = "Failed to generate a summary for the comments."

        except Exception as e:
            error = f"An unexpected error occurred during comment summarization: {e}"
            print(f"Error in summarize_comments_route: {e}")

    return render_template("index.html",
                           chat_history=chat_history,
                           video_url=video_url,
                           previous_video_url=previous_video_url,
                           error=error)


if __name__ == "__main__":
    # Use 0.0.0.0 to be accessible from outside the container if needed
    # Use debug=True for development (auto-reloads), but turn off for production
    app.run(host='0.0.0.0', port=5001, debug=True)
