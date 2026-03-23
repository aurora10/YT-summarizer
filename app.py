import uuid
import yt_dlp
import urllib.request
import json
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import re
import time

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
# Secure random key for session cookie signing
app.secret_key = os.urandom(24)
CORS(app)

# In-memory session store. Structured as:
# { "uuid": {"chat_session": ChatSession, "video_url": str, "transcript": str, "lang_code": str} }
SESSIONS = {}

def get_session_data(uid):
    if uid not in SESSIONS:
        SESSIONS[uid] = {"chat_session": None, "video_url": None, "transcript": None, "lang_code": None}
    return SESSIONS[uid]

def fetch_and_parse_subs(sub_list):
    for sub in sub_list:
        if sub.get('ext') == 'json3':
            url = sub['url']
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())
            text_chunks = []
            for event in data.get('events', []):
                if 'segs' in event:
                    for seg in event['segs']:
                        text_chunks.append(seg.get('utf8', ''))
            val = "".join(text_chunks).replace('\n', ' ').replace('  ', ' ').strip()
            if val: return val
            
    for sub in sub_list:
        if sub.get('ext') == 'vtt':
            url = sub['url']
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                data = response.read().decode('utf-8')
            lines = data.split('\n')
            text_chunks = []
            for line in lines:
                if '-->' in line or line.startswith('WEBVTT') or line.strip() == '' or line.startswith('Kind:') or line.startswith('Language:') or line.startswith('Style:'):
                    continue
                clean_line = re.sub(r'<[^>]+>', '', line)
                if clean_line.strip():
                    text_chunks.append(clean_line.strip())
            val = " ".join(text_chunks).replace('  ', ' ').strip()
            if val: return val
    return None

def yt_dlp_fallback(youtube_url, target_langs):
    cookie_path = os.path.join(os.path.dirname(__file__), 'cookies.txt')
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }
    if os.path.exists(cookie_path):
        ydl_opts['cookiefile'] = cookie_path
        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        
    subs = info.get('subtitles', {})
    auto_subs = info.get('automatic_captions', {})
    
    for lang in target_langs:
        if lang in subs:
            parsed = fetch_and_parse_subs(subs[lang])
            if parsed: return parsed, lang
            
    for lang in target_langs:
        if lang in auto_subs:
            parsed = fetch_and_parse_subs(auto_subs[lang])
            if parsed: return parsed, lang
            
    if subs:
        lang = next(iter(subs))
        parsed = fetch_and_parse_subs(subs[lang])
        if parsed: return parsed, lang
        
    if auto_subs:
        lang = next(iter(auto_subs))
        parsed = fetch_and_parse_subs(auto_subs[lang])
        if parsed: return parsed, lang
        
    return None, None

def extract_transcript(youtube_url):
    try:
        match = re.search(r'(?:https?:\/\/)?(?:[a-zA-Z0-9_-]+\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})', youtube_url)
        if match:
            video_id = match.group(1)
        else:
            return None, "Invalid YouTube URL format."
        
        # Support both old v0.x and new v1.x API paradigms
        if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            transcript_list = YouTubeTranscriptApi().list(video_id)

        target_langs = ['en', 'ru', 'fr']
        transcript = None

        try:
            transcript = transcript_list.find_generated_transcript(target_langs)
        except Exception:
            pass

        if not transcript:
            try:
                transcript = transcript_list.find_manually_created_transcript(target_langs)
            except Exception:
                pass

        if not transcript:
            try:
                transcript = next(iter(transcript_list))
            except StopIteration:
                return None, f"No transcripts available at all for video: {youtube_url}"
            except Exception as e:
                return None, f"Error getting fallback transcript: {e}"

        if not transcript:
            yt_text, yt_lang = yt_dlp_fallback(youtube_url, target_langs)
            if yt_text: return yt_text, yt_lang
            return None, f"No suitable transcripts found for video: {youtube_url}"

        lang_code = transcript.language_code
        try:
            fetched_transcript = transcript.fetch()
        except Exception as e:
            yt_text, yt_lang = yt_dlp_fallback(youtube_url, target_langs)
            if yt_text: return yt_text, yt_lang
            return None, f"Error fetching transcript details: {e}"

        processed_entries = []
        if fetched_transcript:
            for entry in fetched_transcript:
                if hasattr(entry, 'text'):
                    processed_entries.append(entry.text)
                elif isinstance(entry, dict) and 'text' in entry: # in case it's dict
                    processed_entries.append(entry['text'])
            transcript_text = " ".join(processed_entries)
        else:
            yt_text, yt_lang = yt_dlp_fallback(youtube_url, target_langs)
            if yt_text: return yt_text, yt_lang
            return None, "Transcript was empty."

        return transcript_text, lang_code

    except yt_dlp.utils.DownloadError as e:
        return None, f"yt-dlp error: {e}"
    except Exception as e:
        try:
            yt_text, yt_lang = yt_dlp_fallback(youtube_url, ['en', 'ru', 'fr'])
            if yt_text: return yt_text, yt_lang
        except Exception:
            pass
        return None, f"An error occurred: {type(e).__name__}: {e}"

def clean_transcript(transcript):
    profanities = []
    if not profanities:
        return transcript
    cleaned_transcript = re.sub(
        r'\b(' + '|'.join(profanities) + r')\b', '[REMOVED]', transcript, flags=re.IGNORECASE)
    return cleaned_transcript

def send_message_with_retry(chat, message, max_retries=3, backoff_delay=1):
    retries = 0
    while retries <= max_retries:
        try:
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                retries += 1
                if retries > max_retries:
                    return "The service is currently busy due to rate limits. Please try again later."
                time.sleep(backoff_delay * retries)
            else:
                return f"An error occurred: {str(e)}"
    return "Error communicating with LLM."

@app.route("/")
def index():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/api/process_video", methods=["POST"])
def process_video():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "No session found"}), 400
        
    data = request.get_json()
    video_url = data.get("video_url", "").strip()
    
    if not video_url:
        return jsonify({"error": "Video URL is required"}), 400
        
    session_data = get_session_data(user_id)
    
    transcript_text, lang_code = extract_transcript(video_url)
    
    if transcript_text is None:
        return jsonify({"error": lang_code}), 400
        
    cleaned_transcript = clean_transcript(transcript_text)
    
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return jsonify({"error": "Google API Key missing. Please set it in .env file."}), 500
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        chat = model.start_chat(history=[])
        
        system_prompt = f"You are a helpful assistant. The language of the video transcript is '{lang_code}'. Please respond and analyze in the same language. Provide a comprehensive summary using bullet points. If the user asks follow-up questions, use your knowledge if the answer is not in the transcript.\n\n--- Video Transcript ---\n{cleaned_transcript}\n--- End Transcript ---"
        
        summary = send_message_with_retry(chat, system_prompt)
        
        # Save session context so we don't have to re-fetch on follow-ups
        session_data["video_url"] = video_url
        session_data["chat_session"] = chat
        session_data["transcript"] = cleaned_transcript
        session_data["lang_code"] = lang_code
        
        return jsonify({"summary": summary})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "No session found. Please refresh the page."}), 400
        
    data = request.get_json()
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
        
    session_data = get_session_data(user_id)
    chat_session = session_data.get("chat_session")
    
    if not chat_session:
        # Fallback if chat session drops, or if general chat without video
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            chat_session = model.start_chat(history=[])
            session_data["chat_session"] = chat_session
        except Exception as e:
            return jsonify({"error": "Error initializing general chat context."}), 500
            
    response = send_message_with_retry(chat_session, user_message)
    return jsonify({"response": response})

@app.route("/api/clear", methods=["POST"])
def clear():
    user_id = session.get("user_id")
    if user_id in SESSIONS:
        SESSIONS[user_id] = {"chat_session": None, "video_url": None, "transcript": None, "lang_code": None}
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
