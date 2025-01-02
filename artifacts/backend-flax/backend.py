# backend.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/chat", methods=["POST"])
def chat_endpoint():
     data = request.get_json()
     transcript = data.get("transcript")
     user_input = data.get("user_input")
     conversation_history = data.get("conversation_history", []) # Ensure conversation history is passed in

     if not transcript or not user_input:
         return jsonify({"error": "Missing transcript or user input"}), 400

     try:
         api_key = os.environ.get("GOOGLE_API_KEY")
         if not api_key:
             raise ValueError("API key not found. Ensure you have set the GOOGLE_API_KEY environment variable or passed it directly.")
         
         genai.configure(api_key=api_key)
         model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp')

         prompt = f"""
            You are an expert at answering questions about a given text and will provide helpful answers to the user.
            Here is the transcript of the video:\n{transcript}\n
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
           conversation_history.append(f"User: {user_input}\nLLM: {llm_response}") # Update conversation history
           return jsonify({"llm_response": llm_response, "conversation_history": conversation_history})
         else:
            return jsonify({"error": "LLM did not generate a response"}), 500

     except Exception as e:
        return jsonify({"error": f"An exception occurred: {type(e)} {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)