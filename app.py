import yt_dlp
# Import specific exceptions
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import os
from dotenv import load_dotenv
# from datetime import datetime # Not used currently, can be removed if not needed elsewhere
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import markdown
import re
# from langdetect import detect # Not used currently, can be removed if not needed elsewhere
import time

# Load environment variables from .env file if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes.


def extract_transcript(youtube_url):
    """
    Extracts and returns the transcript of a YouTube video, prioritizing English, Russian, and French.

    Args:
        youtube_url: The URL of the YouTube video.

    Returns:
        A tuple containing:
           - A string representing the transcript or None if no suitable transcript is found
           - A string representing the language code of the transcript (e.g., 'en', 'ru', 'fr') or None
           OR if an error occurs:
           - None
           - A string containing the error message
    """
    try:
        video_id = yt_dlp.YoutubeDL({}).extract_info(  # Pass empty dict to avoid potential config issues
            youtube_url, download=False)['id']
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prioritize generated transcripts first, then manual
        available_langs = [t.language_code for t in transcript_list]
        print(f"Available transcript languages: {available_langs}")

        target_langs = ['en', 'ru', 'fr']  # Your preferred languages

        transcript = None
        # Try finding a generated transcript in preferred languages
        try:
            transcript = transcript_list.find_generated_transcript(
                target_langs)
        except Exception:
            print(
                f"Could not find generated transcript in {target_langs}, checking manual...")

        # If not found, try finding a manually created transcript in preferred languages
        if not transcript:
            try:
                transcript = transcript_list.find_manually_created_transcript(
                    target_langs)
            except Exception:
                print(
                    f"Could not find manual transcript in {target_langs}, checking any available...")

        # If still not found, try *any* available transcript as a last resort
        # (You might want to remove this block if you ONLY want en/ru/fr)
        if not transcript:
            try:
                # Pick the first available transcript regardless of language
                transcript = next(iter(transcript_list))
                print(
                    f"Falling back to available language: {transcript.language}")
            except StopIteration:
                print(
                    f"No transcripts available at all for video: {youtube_url}")
                # Return None, specific message or None for generic handling below
                return None, f"No transcripts available at all for video: {youtube_url}"
            except Exception as e:
                print(f"Error getting fallback transcript: {e}")
                # Return specific error
                return None, f"Error getting fallback transcript: {e}"

        if not transcript:
            print(f"No suitable transcripts found for video: {youtube_url}")
            # Return specific error
            return None, f"No suitable transcripts found for video: {youtube_url}"

        # This block should be correctly indented within the 'try'
        print(
            f"Selected transcript language: {transcript.language} ({transcript.language_code})")

        # Isolate the fetch call
        fetched_transcript = None
        try:
            fetched_transcript = transcript.fetch()
            print(
                f"Fetched transcript type: {type(fetched_transcript)}, length: {len(fetched_transcript) if isinstance(fetched_transcript, list) else 'N/A'}")
        except AttributeError as fetch_ae:
            print(f"AttributeError during transcript.fetch(): {fetch_ae}")
            # Return specific error related to fetching details
            return None, f"Error fetching transcript details: {fetch_ae}"
        except Exception as fetch_e:
            print(
                f"Unexpected error during transcript.fetch(): {type(fetch_e).__name__}: {fetch_e}")
            return None, f"Unexpected error fetching transcript details: {type(fetch_e).__name__}"

        # Explicit loop for processing entries, only if fetch was successful
        processed_entries = []
        transcript_text = ""  # Default to empty string
        if fetched_transcript:
            # Correctly indented loop
            for entry in fetched_transcript:
                # print(f"Processing entry: {entry}, type: {type(entry)}") # Optional detailed debug
                # Check if the entry object has a 'text' attribute
                if hasattr(entry, 'text'):
                    # Access using attribute
                    processed_entries.append(entry.text)
                else:
                    # Log a warning if the format is unexpected (neither dict nor object with text attr)
                    print(
                        f"Warning: Skipping unexpected transcript entry format: {entry}")
            # Join the text *after* the loop finishes
            transcript_text = " ".join(processed_entries)

            # Check for empty extraction *after* the loop
            if not transcript_text and fetched_transcript:
                # Add warning if extraction failed
                print(
                    "Warning: Transcript fetched but no text could be extracted from entries.")
        else:
            # Handle case where fetch failed or returned None/empty
            print(
                "Warning: Fetched transcript was empty or fetch failed, proceeding without text.")

        # Get lang_code regardless of fetch success (might still be useful)
        lang_code = transcript.language_code

        return transcript_text, lang_code

    except yt_dlp.utils.DownloadError as e:
        print(f"Error fetching video information with yt-dlp: {e}")
        # Return a more specific error message if possible
        return None, f"yt-dlp error: {e}"
    except TranscriptsDisabled as e:
        video_id = e.video_id  # Get video ID from exception
        print(f"Transcripts are disabled for video {video_id}: {e}")
        return None, f"Transcripts are disabled for this video ({video_id})."
    except NoTranscriptFound as e:
        video_id = e.video_id
        print(
            f"No transcript found for video {video_id} in requested languages: {e}")
        # You might want to list available languages if the exception provides them
        return None, f"No suitable transcript found for this video ({video_id})."
    except Exception as e:
        # Catch any other unexpected errors
        print(
            f"An unexpected error occurred getting transcript: {type(e).__name__}: {e}")
        return None, f"An unexpected error occurred: {type(e).__name__}"


def clean_transcript(transcript):
    """Simple profanity filter (you may need to expand this)."""
    # Example profanity list, you should expand it
    profanities = []  # This should be expanded
    if not profanities:  # Avoid regex error with empty list
        return transcript
    cleaned_transcript = re.sub(
        r'\b(' + '|'.join(profanities) + r')\b', '[REMOVED]', transcript, flags=re.IGNORECASE)
    return cleaned_transcript


# ----- MODIFIED FUNCTION -----
def chat_with_llm(text, user_input, conversation_history, lang_code, api_key=None, max_retries=3, backoff_delay=1):
    """Starts a conversational interaction with the LLM, enforcing the specified language."""
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found.")

    # Default to English if lang_code is missing for some reason
    effective_lang_code = lang_code if lang_code else "en"
    # Add logging
    print(f"LLM instructed to respond in: {effective_lang_code}")

    try:
        genai.configure(api_key=api_key)
        # Consider using gemini-1.5-flash if available/preferred
        model = genai.GenerativeModel(model_name='gemini-2.0-flash')

        retries = 0
        while retries <= max_retries:
            try:
                # --- Prompt Modification ---
                # Construct the core instruction using the effective language code
                language_instruction = f"**IMPORTANT: You MUST respond ONLY in the language identified by the code: {effective_lang_code}.** Do not use any other language."

                # Base prompt structure
                prompt_parts = [
                    "You are a helpful assistant."
                ]

                # Add transcript-specific instructions if transcript text exists
                if text:
                    prompt_parts.extend([
                        f"The language of the video transcript is '{effective_lang_code}'.",
                        "Summarize the video transcript using bulet points when fits context and answer the user's question.",
                        language_instruction,  # Reinforce language constraint
                        "\n--- Video Transcript ---",
                        text,
                        "--- End Transcript ---"
                    ])
                else:
                    # If no transcript, just focus on the chat and language
                    prompt_parts.extend([
                        language_instruction,
                        "Answer the user's question based on the chat history."
                    ])

                # Add chat history and user input
                prompt_parts.extend([
                    "\n--- Chat History ---",
                    conversation_history,  # Assuming conversation_history is already a string
                    "--- End History ---",
                    f"\nUser: {user_input}",
                    f"\nFormat your response in Markdown.",
                    # Hint the expected output language
                    f"LLM ({effective_lang_code}):"
                ])

                prompt = "\n".join(prompt_parts)
                # print(f"\n--- Sending Prompt to LLM ---\n{prompt}\n--- End Prompt ---") # Optional: for debugging

                response = model.generate_content(prompt, safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                })

                # --- Response Handling (More Robust) ---
                # Check if response has text directly (common case)
                if hasattr(response, 'text'):
                    return response.text

                # Check candidates if direct text isn't available (older API versions or complex responses)
                elif hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    # Check for valid content part
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # Check safety finish reason
                        # 1 usually means STOP (normal)
                        if hasattr(candidate, "finish_reason") and candidate.finish_reason == 1:
                            return candidate.content.parts[0].text
                        elif hasattr(candidate, "finish_reason") and candidate.finish_reason != 1:
                            print(
                                f"LLM generation stopped for reason: {candidate.finish_reason}. Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}")
                            # Provide a more informative message based on finish reason if possible
                            if candidate.finish_reason == 3:  # SAFETY
                                return "My safety filters prevented me from generating a response to this request."
                            else:
                                return f"The response generation stopped unexpectedly (Reason: {candidate.finish_reason})."
                        else:
                            # If finish_reason missing but content exists, cautiously return it
                            print(
                                "Warning: Finish reason missing, but content found.")
                            return candidate.content.parts[0].text
                    else:
                        # If candidate exists but has no valid content
                        print(
                            f"LLM response candidate lacked valid content parts. Finish Reason: {getattr(candidate, 'finish_reason', 'N/A')}")
                        return "The LLM generated an empty or invalid response structure."

                # If response object structure is unexpected
                else:
                    print(f"Unexpected LLM response structure: {response}")
                    return "The LLM returned an unexpected response format."

            except Exception as e:
                # Check for specific RateLimitError if the SDK provides it
                # from google.api_core.exceptions import ResourceExhausted # Example, check actual exception type
                # if isinstance(e, ResourceExhausted) or "429" in str(e): # Check specific error type if possible
                # More generic check for rate limits
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    retries += 1
                    if retries > max_retries:
                        print("Max retries exceeded due to rate limiting.")
                        return "The service is currently busy. Please try again later."
                    print(
                        f"Rate limit hit. Retrying in {backoff_delay * retries}s... (Attempt {retries}/{max_retries})"
                    )
                    time.sleep(backoff_delay * retries)  # Exponential backoff
                else:
                    # Handle other potential errors
                    print(
                        f"An unexpected error occurred during LLM generation: {type(e).__name__}: {e}")
                    # You might want to return a more user-friendly error here
                    return f"An error occurred while generating the response. Details: {type(e).__name__}"

        # This line is reached if all retries fail (likely due to rate limits)
        return "Max retries exceeded. The service might be temporarily unavailable or overloaded."

    except Exception as e:
        # Catch errors during initial setup (e.g., API key config)
        print(
            f"An exception occurred configuring or calling the LLM: {type(e).__name__}: {e}")
        return f"An error occurred: {type(e).__name__}"

# ----- END MODIFIED FUNCTION -----


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    chat_history = ""  # Stores the HTML formatted history for display
    video_url = ""
    previous_video_url = ""  # Keep track of the URL processed in the last request

    if request.method == "POST":
        video_url = request.form.get(
            "video_url", "").strip()  # Get URL from main input
        user_message = request.form.get("user_message", "").strip()
        # Get current history from hidden input
        chat_history = request.form.get("chat_history", "")
        clear_chat = request.form.get("clear_chat")
        # Get URL associated with the current chat history from hidden input
        previous_video_url = request.form.get("previous_video_url", "")

        # --- Input Handling Logic ---
        if clear_chat:
            # Clear everything and render empty form
            return render_template("index.html", error=None, chat_history="", video_url="", previous_video_url="")

        if user_message and user_message.lower() == "exit":
            # Treat "exit" like clearing the chat
            return render_template("index.html", error=None, chat_history="", video_url="", previous_video_url="")

        # Check if a new video URL has been entered
        is_new_video = bool(video_url) and (video_url != previous_video_url)

        # --- Processing Logic ---
        transcript = None
        lang_code = None
        response_text = None  # Initialize response_text

        if is_new_video:
            print(f"New video URL detected: {video_url}")
            chat_history = ""  # Reset chat history for new video
            previous_video_url = video_url  # Update the previous URL tracker
            if not user_message:
                user_message = "Summarize this video"  # Default action for new video
            # Extract transcript for the new video
            transcript_text, error_message_or_lang_code = extract_transcript(
                video_url)
            if transcript_text is None:
                # Use the more specific error message from the function
                error = error_message_or_lang_code or f"Could not extract transcript for {video_url}. Please check the URL and ensure transcripts are available."
                # Keep video_url in the input box, but clear previous_video_url as processing failed
                previous_video_url = ""
                lang_code = None  # Ensure lang_code is None on error
            else:
                # Success case
                transcript = transcript_text
                lang_code = error_message_or_lang_code  # This is the lang_code on success
                cleaned_transcript = clean_transcript(transcript)
                # Call LLM with the transcript and user message
                # Pass empty history for summary
                response_text = chat_with_llm(
                    cleaned_transcript, user_message, "", lang_code)

        elif user_message and previous_video_url:
            # Continue chat for the *previous* video
            print(f"Continuing chat for video: {previous_video_url}")
            # We need the transcript again for context in chat_with_llm
            # Re-extract transcript (consider caching in a real app)
            transcript_text, error_message_or_lang_code = extract_transcript(
                previous_video_url)
            if transcript_text is None:
                # Use the more specific error message
                error = error_message_or_lang_code or f"Could not re-fetch transcript for context ({previous_video_url}). Please try entering the URL again."
                # Clear state as we lost context
                chat_history = ""
                video_url = previous_video_url  # Keep the URL that failed in the box
                previous_video_url = ""  # Clear history association
                lang_code = None  # Ensure lang_code is None on error
            else:
                # Success case
                transcript = transcript_text
                lang_code = error_message_or_lang_code  # This is the lang_code on success
                cleaned_transcript = clean_transcript(transcript)
                # Note: Passing the *raw* chat_history string. chat_with_llm expects this format.
                # Consider cleaning the HTML tags from chat_history before sending if the LLM gets confused.
                response_text = chat_with_llm(
                    cleaned_transcript, user_message, chat_history, lang_code)

        elif user_message and not previous_video_url:
            # General chat, no video context
            print("General chat, no video context.")
            # Pass empty transcript, empty history, default lang to 'en'
            response_text = chat_with_llm("", user_message, chat_history, "en")
            # Ensure video_url and previous_video_url remain empty for general chat state
            video_url = ""
            previous_video_url = ""

        # --- Update Chat History ---
        if response_text:
            # Append user message and LLM response to the HTML history
            chat_history += f'<div class="message user-message"><span class="role">You:</span> {user_message}</div>\n'
            # Use markdown.markdown for formatting LLM response
            formatted_response = markdown.markdown(response_text)
            chat_history += f'<div class="message llm-message"><span class="role">LLM:</span> {formatted_response}</div>\n'

        # --- Render Template ---
        # Pass the current state back to the template
        return render_template("index.html",
                               chat_history=chat_history,
                               video_url=video_url,  # URL currently in the input box
                               previous_video_url=previous_video_url,  # URL associated with the chat history
                               error=error)

    # Initial GET request or if POST logic doesn't render
    return render_template("index.html", error=None, chat_history="", video_url="", previous_video_url="")


if __name__ == "__main__":
    # Use 0.0.0.0 to be accessible from outside the container if needed
    # Use debug=True for development (auto-reloads), but turn off for production
    app.run(host='0.0.0.0', port=5001, debug=True)
