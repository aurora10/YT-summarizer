import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import markdown

# Load environment variables from .env file if present
#load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))


app = Flask(__name__)
CORS(app) # Enable CORS for all routes.


def extract_transcript(youtube_url):
    """
    Extracts and returns the transcript of a YouTube video.

    Args:
        youtube_url: The URL of the YouTube video.

    Returns:
        A string containing the transcript or None if no transcript is found or
        an error occurs.
    """

    try:
        video_id = yt_dlp.YoutubeDL().extract_info(youtube_url, download=False)['id']
        
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en'])  # Prioritize generated English transcript
        
        transcript_text = " ".join([entry['text'] for entry in transcript.fetch()])
        return transcript_text

    except yt_dlp.utils.DownloadError as e:
       print(f"Error fetching video information: {e}")
       return None
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None


def chat_with_llm(text, user_input, conversation_history, api_key=None):
    """
    Starts a conversational interaction with the LLM using the given video transcript.

    Args:
        text (str): The video transcript.
        user_input (str): The user message.
        conversation_history (list): The chat history.
        api_key (str, optional): Your Gemini API key. Defaults to None and reads from the environment.

    Returns:
      A string which is the LLM output.
    """

    if not api_key:
      api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
      raise ValueError("API key not found. Ensure you have set the GOOGLE_API_KEY environment variable or passed it directly.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp')

        # Formulate the prompt including conversation history
        prompt = f"""
            You are an expert at answering questions about a given text and will provide helpful answers to the user. 
            Here is the transcript of the video: \n{text}\n Summarize it first.
            Here is the chat history:\n{conversation_history}\n
            User: {user_input}\n
            Format your response in Markdown.
            LLM:
        """
        response = model.generate_content(prompt, safety_settings={
              HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
              HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
              HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
              HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
          })
        if hasattr(response, "text") and response.text:
           llm_response = response.text
           return llm_response
        else:
           return "The LLM didn't return an answer."
    except Exception as e:
        print(f"An exception occurred: {type(e)} {e}")
        return f"Error: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    chat_history = ""
    video_url = ""


    if request.method == "POST":
        video_url = request.form.get("video_url", "")
        user_message = request.form.get("user_message")
        chat_history = request.form.get("chat_history", "") # set default to empty string
        clear_chat = request.form.get("clear_chat")
        previous_video_url = request.form.get("previous_video_url", "")

        if clear_chat:
          return render_template("index.html", error=error, chat_history="")
        

        if user_message and user_message.lower() == "exit":
           return render_template("index.html", error=error, chat_history="") # clears the chat history
       
        if video_url and previous_video_url != video_url:
           chat_history="" #clears chat history when a new video is introduced
           if not user_message: # add summary only if user_message was not defined
             user_message = "Summarize this video" # PREPEND MESSAGE HERE

        if video_url:
            transcript = extract_transcript(video_url)
            if transcript:
                if user_message:
                    chat_history_list = chat_history.split("<br>")
                    response = chat_with_llm(transcript, user_message, chat_history_list)
                    chat_history += f'<span class="user-prompt">You:</span> {user_message}<br><span class="llm-response">LLM:</span> {markdown.markdown(response)}<br>' # Adding a span here
                return render_template("index.html", chat_history=chat_history, video_url=video_url, previous_video_url=video_url)
            else:
                 error = "Could not extract transcript"
        else:
           if user_message and chat_history: # Continue the chat if video_url was not passed, but chat_history and user_message was
               chat_history_list = chat_history.split("<br>")
               # We need to pass a dummy URL to make sure the video can be extracted
               response = chat_with_llm("", user_message, chat_history_list)
               chat_history += f'<span class="user-prompt">You:</span> {user_message}<br><span class="llm-response">LLM:</span> {markdown.markdown(response)}<br>'
               return render_template("index.html", chat_history=chat_history, video_url=request.form.get("video_url_initial", ""), previous_video_url=request.form.get("previous_video_url", "")) # we also add video_url_initial to keep video url in the input field

    return render_template("index.html", error=error, chat_history=chat_history, video_url=video_url)


if __name__ == "__main__":
        app.run(port=5001, host='127.0.0.1')