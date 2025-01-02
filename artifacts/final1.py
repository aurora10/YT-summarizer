import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables from .env file if present
load_dotenv()

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


def chat_with_llm(text, api_key=None):
    """
    Starts a conversational interaction with the LLM using the given video transcript.

    Args:
        text (str): The video transcript.
        api_key (str, optional): Your Gemini API key. Defaults to None and reads from the environment.

    Returns:
      None
    """

    if not api_key:
      api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
      raise ValueError("API key not found. Ensure you have set the GOOGLE_API_KEY environment variable or passed it directly.")


    try:
      genai.configure(api_key=api_key)
      model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp')
      conversation_history = [] # Initialize conversation history

      print("LLM chat initialized. Type 'exit' to end the conversation.")

      while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                  print("Chat session ended.")
                  break
            
            # Formulate the prompt including conversation history
            prompt = f"""
            You are an expert at answering questions about a given text and will provide helpful answers to the user. 
            Here is the transcript of the video: \n{text}\n Summarize it first.
            Here is the chat history:\n{conversation_history}\n
            User: {user_input}\n
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
               print(f"LLM: {llm_response}")
                # Save user message and LLM response to conversation history
               conversation_history.append(f"User: {user_input}\nLLM: {llm_response}")
            else:
                print("The LLM didn't return an answer.")

    except Exception as e:
      print(f"An exception occurred: {type(e)} {e}")



if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    transcript = extract_transcript(video_url)

    if transcript:
        chat_with_llm(transcript)
    else:
        print("Could not extract transcript or an error occurred.")