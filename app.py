import yt_dlp
import json  # Required for parsing transcript JSON

from googleapiclient.discovery import build
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import markdown
import re
import time

# Load environment variables from .env file if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes.

# Create an API key from your Google Cloud Console
# Using the provided key for both YouTube Data API and Gemini API for now.
# In a production environment, these should ideally be separate and loaded from environment variables.
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Create a directory named 'Comments' if it doesn't exist
comments_dir = "Comments"
if not os.path.exists(comments_dir):
    os.makedirs(comments_dir)


def get_comments(video_id):
    """Fetch comments from YouTube Data API."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []

    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        if 'nextPageToken' in results:
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=results['nextPageToken'],
                textFormat="plainText",
                maxResults=100
            ).execute()
        else:
            break
    return comments


def extract_transcript(youtube_url):
    """
    Extracts YouTube video transcript using yt-dlp with robust error handling.

    Args:
        youtube_url: YouTube video URL

    Returns:
        Tuple: (transcript_text, lang_code) or (None, error_message)
    """
    try:
        # Configure yt-dlp options for transcript extraction
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'subtitleslangs': ['en', 'ru', 'fr', 'auto'],
            'subtitlesformat': 'json3',
            'quiet': False,
            'no_warnings': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_id = info['id']
            print(f"Video ID: {video_id}")

            # Extract available caption languages
            available_captions = info.get('automatic_captions', {})
            if not available_captions:
                available_captions = info.get('subtitles', {})

            available_langs = list(available_captions.keys())
            print(f"Available transcript languages: {available_langs}")

            # Try preferred languages in order
            target_langs = ['en', 'ru', 'fr', 'auto']
            transcript_text = None
            lang_code = None

            for lang in target_langs:
                if lang in available_captions:
                    caption_data = available_captions[lang]
                    if caption_data:
                        # Get the first caption format (usually json3)
                        caption_url = caption_data[0]['url']
                        # Fetch caption content
                        transcript_text = ydl.urlopen(
                            caption_url).read().decode('utf-8')

                        # Parse JSON3 format
                        try:
                            transcript_json = json.loads(transcript_text)
                            events = transcript_json.get('events', [])
                            segments = []
                            for event in events:
                                if 'segs' in event:
                                    for seg in event['segs']:
                                        if 'utf8' in seg:
                                            segments.append(seg['utf8'])
                            transcript_text = " ".join(segments)
                            lang_code = lang
                            print(f"Using transcript language: {lang}")
                            break
                        except Exception as e:
                            print(f"Error parsing transcript JSON: {e}")
                            continue

            if transcript_text:
                return transcript_text, lang_code
            else:
                return None, "No suitable transcript found for this video"

    except yt_dlp.utils.DownloadError as e:
        return None, f"yt-dlp error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {type(e).__name__}: {e}"


def clean_transcript(transcript):
    """Simple profanity filter (you may need to expand this)."""
    return transcript


# ----- MODIFIED FUNCTION -----
def chat_with_llm(text, user_input, conversation_history, lang_code, api_key=None, max_retries=3, backoff_delay=1):
    """Starts a conversational interaction with the LLM, enforcing the specified language."""
    if not api_key:
        api_key = GOOGLE_API_KEY  # Use the globally defined GOOGLE_API_KEY
    if not api_key:
        raise ValueError("API key not found.")

    # Default to English if lang_code is missing for some reason
    effective_lang_code = lang_code if lang_code else "en"
    # Add logging
    print(f"LLM instructed to respond in: {effective_lang_code}")

    try:
        genai.configure(api_key=api_key)
        # Consider using gemini-1.5-flash if available/preferred
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash')
        # gemini-2.5-pro-preview-05-06 gemini-2.0-flash

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
                        "Summarize the video transcript using bulet points when fits context and answer the user's question. If user asks follow up question and the answer in not found in transcript, then use your own knowlege to answer the follow up question.",
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
                # print(f"\n--- Sending Prompt to LLM---\n{prompt}\n--- End Prompt ---") # Optional: for debugging

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
                        f"Rate limit hit. Retrying in {backoff_delay * retries}s... (Attempt {retries}/{max_retries})")
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

# ----- NEW FUNCTION FOR QUESTION GENERATION -----


def generate_questions_from_transcript(transcript_text, lang_code, api_key=None, max_retries=3, backoff_delay=1):
    """
    Generates 4 concise questions (3-4 words) about the video content using Gemini LLM.
    Returns only the question text without numbering or explanations.

    Args:
        transcript_text: Cleaned transcript text
        lang_code: Language code of the transcript
        api_key: Gemini API key
        max_retries: Number of retries for API errors
        backoff_delay: Delay between retries

    Returns:
        List of 4 questions or None on failure
    """
    if not api_key:
        api_key = GOOGLE_API_KEY  # Use the globally defined GOOGLE_API_KEY
    if not api_key:
        print("API key not found for question generation.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-2.0-flash')

        # Create specialized prompt
        prompt = (
            f"**IMPORTANT: You MUST respond ONLY in the language identified by the code: {lang_code}.**\n"
            "Analyze the following video transcript and generate exactly 4 concise questions "
            "that viewers might have about this video. Each question MUST be 3-4 words maximum. "
            "Format your response as a simple numbered list (1. 2. 3. 4.):\n"
            "--- TRANSCRIPT START ---"
            f"{transcript_text}\n"
            "--- TRANSCRIPT END ---"
        )

        response = model.generate_content(prompt, safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        })

        if hasattr(response, 'text'):
            # Extract only the numbered list items
            lines = response.text.strip().split('\n')
            questions = []
            for line in lines:
                # Match lines that start with a number and period
                if re.match(r'^\d+\.\s', line):
                    # Remove numbering and trim
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    questions.append(question)
            return questions[:4]  # Return up to 4 valid questions
        return None

    except Exception as e:
        print(f"Question generation error: {e}")
        return None
# ----- END NEW FUNCTION -----

# ----- NEW FUNCTION FOR SUMMARIZING COMMENTS -----


def summarize_text_with_llm(text_to_summarize, prompt_instruction, api_key=None, max_retries=3, backoff_delay=1):
    """
    Summarizes provided text using the Gemini LLM based on a given instruction.
    """
    if not api_key:
        api_key = GOOGLE_API_KEY
    if not api_key:
        raise ValueError("API key not found for summarization.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-2.0-flash')

        full_prompt = f"""
        {prompt_instruction}

        Text to summarize:
        {text_to_summarize}
        """

        retries = 0
        while retries <= max_retries:
            try:
                response = model.generate_content(full_prompt, safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                })

                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                        if hasattr(candidate, "finish_reason") and candidate.finish_reason == 1:
                            return candidate.content.parts[0].text
                        else:
                            print(
                                f"LLM generation stopped for reason: {candidate.finish_reason}. Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}")
                            if candidate.finish_reason == 3:  # SAFETY
                                return "My safety filters prevented me from generating a response to this request."
                            else:
                                return f"The response generation stopped unexpectedly (Reason: {candidate.finish_reason})."
                    else:
                        print(
                            f"LLM response candidate lacked valid content parts. Finish Reason: {getattr(candidate, 'finish_reason', 'N/A')}")
                        return "The LLM generated an empty or invalid response structure."
                else:
                    print(f"Unexpected LLM response structure: {response}")
                    return "The LLM returned an unexpected response format."
            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    retries += 1
                    if retries > max_retries:
                        print("Max retries exceeded due to rate limiting.")
                        return "The service is currently busy. Please try again later."
                    print(
                        f"Rate limit hit. Retrying in {backoff_delay * retries}s... (Attempt {retries}/{max_retries})")
                    time.sleep(backoff_delay * retries)
                else:
                    print(
                        f"An unexpected error occurred during LLM generation: {type(e).__name__}: {e}")
                    return f"An error occurred while generating the response. Details: {type(e).__name__}"
        return "Max retries exceeded. The service might be temporarily unavailable or overloaded."

    except Exception as e:
        print(
            f"An exception occurred configuring or calling the LLM: {type(e).__name__}: {e}")
        return f"An error occurred: {type(e).__name__}"
# ----- END NEW FUNCTION FOR SUMMARIZING COMMENTS -----


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    chat_history = ""  # Stores the HTML formatted history for display
    video_url = ""
    previous_video_url = ""  # Keep track of the URL processed in the last request
    questions = []  # Initialize questions to an empty list

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

                # Generate and print questions for new video
                print("\n" + "="*50)
                print("Generating questions about this video...")
                try:
                    questions = generate_questions_from_transcript(
                        cleaned_transcript, lang_code)
                    if questions:
                        print("\nGenerated Questions:")
                        for i, q in enumerate(questions, 1):
                            print(f"{i}. {q}")
                    else:
                        print("Question generation failed")
                except Exception as e:
                    print(f"Error generating questions: {e}")
                print("="*50 + "\n")

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
            # Add language metadata to LLM responses
            chat_history += f'<div class="message llm-message" lang="{lang_code}"><span class="role">LLM:</span> {formatted_response}</div>\n'

        # --- Render Template ---
        # Pass the current state back to the template
        return render_template("index.html",
                               chat_history=chat_history,
                               video_url=video_url,  # URL currently in the input box
                               previous_video_url=previous_video_url,  # URL associated with the chat history
                               error=error,
                               generated_questions=questions if is_new_video and questions else [])

    # Initial GET request or if POST logic doesn't render
    return render_template("index.html", error=None, chat_history="", video_url="", previous_video_url="")


@app.route("/summarize_comments", methods=["POST"])
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
                r'(?:v=|\/|watch\?v=)([a-zA-Z0-9_-]{11})', video_url)
            if not video_id_match:
                error = "Invalid YouTube URL provided."
            else:
                video_id = video_id_match.group(1)
                print(f"Attempting to fetch comments for video ID: {video_id}")
                comments = get_comments(video_id)

                if not comments:
                    error = "No comments found for this video or an error occurred fetching them."
                else:
                    # Join comments into a single string for summarization
                    comments_text = "\n".join(comments)
                    print(f"Fetched {len(comments)} comments. Summarizing...")
                    prompt_instruction = "Summarize the following YouTube comments. Focus on common themes, sentiments, and key discussion points. Provide a concise summary in bullet points if appropriate."
                    summary_text = summarize_text_with_llm(
                        comments_text, prompt_instruction)
                    print("Comments summarization complete.")

                    if summary_text:
                        # Add the summary to the chat history
                        chat_history += f'<div class="message user-message"><span class="role">You:</span> Summarize comments for {video_url}</div>\n'
                        formatted_summary = markdown.markdown(summary_text)
                        chat_history += f'<div class="message llm-message"><span class="role">LLM (Comments Summary):</span> {formatted_summary}</div>\n'
                    else:
                        error = "Failed to generate a summary for the comments."

        except Exception as e:
            error = f"An unexpected error occurred during comment summarization: {e}"
            print(f"Error in summarize_comments_route: {e}")

    return render_template("index.html",
                           chat_history=chat_history,
                           video_url=video_url,
                           previous_video_url=previous_video_url,
                           error=error,
                           generated_questions=[])  # No new questions generated on comment summary


if __name__ == "__main__":
    # Use 0.0.0.0 to be accessible from outside the container if needed
    # Use debug=True for development (auto-reloads), but turn off for production
    app.run(host='0.0.0.0', port=5001, debug=True)
