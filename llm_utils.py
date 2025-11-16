import google.generativeai as genai
import os
import time
import re
from google.generativeai.types import HarmCategory, HarmBlockThreshold


def configure_genai():
    """
    Configures the Gemini API key.
    """
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("API key not found.")
    genai.configure(api_key=api_key)


def call_gemini_api(prompt, max_retries=3, backoff_delay=1):
    """
    Calls the Gemini API with a given prompt, handling API key management,
    model initialization, retry logic, and response parsing.
    """
    gemini_model = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
    print(f"DEBUG: call_gemini_api called with model: {gemini_model}")
    print(f"DEBUG: Prompt length: {len(prompt)} characters")

    try:
        model = genai.GenerativeModel(model_name=gemini_model)
        print(f"DEBUG: Model initialized: {gemini_model}")

        retries = 0
        while retries <= max_retries:
            try:
                print(
                    f"DEBUG: Attempting Gemini API call (attempt {retries + 1}/{max_retries + 1})")
                response = model.generate_content(prompt, safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                })
                print("DEBUG: Gemini API call successful")

                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                        if hasattr(candidate, "finish_reason") and candidate.finish_reason == 1:
                            return candidate.content.parts[0].text
                        else:
                            finish_reason = getattr(
                                candidate, 'finish_reason', 'N/A')
                            safety_ratings = getattr(
                                candidate, 'safety_ratings', 'N/A')
                            print(
                                f"LLM generation stopped for reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                            if finish_reason == 3:  # SAFETY
                                return "My safety filters prevented me from generating a response to this request."
                            else:
                                return f"The response generation stopped unexpectedly (Reason: {finish_reason})."
                    else:
                        finish_reason = getattr(
                            candidate, 'finish_reason', 'N/A')
                        print(
                            f"LLM response candidate lacked valid content parts. Finish Reason: {finish_reason}")
                        return "The LLM generated an empty or invalid response structure."
                else:
                    print(f"Unexpected LLM response structure: {response}")
                    return "The LLM returned an unexpected response format."

            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    retries += 1
                    if retries > max_retries:
                        print(
                            f"Max retries exceeded due to rate limiting. Model: {gemini_model}")
                        return f"API rate limit exceeded. The service is currently busy. Please wait a few minutes and try again. (Model: {gemini_model})"
                    wait_time = backoff_delay * \
                        (2 ** retries)  # Exponential backoff
                    print(
                        f"Rate limit hit for model {gemini_model}. Retrying in {wait_time}s... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                elif "quota" in str(e).lower():
                    print(f"API quota exhausted for model {gemini_model}")
                    return "Daily API quota has been exhausted. Please try again tomorrow or consider upgrading your API plan."
                else:
                    print(
                        f"An unexpected error occurred during LLM generation with model {gemini_model}: {type(e).__name__}: {e}")
                    return f"An error occurred while generating the response. Please try again. (Error: {type(e).__name__})"

        return "Max retries exceeded. The service might be temporarily unavailable or overloaded."

    except Exception as e:
        print(
            f"An exception occurred configuring or calling the LLM: {type(e).__name__}: {e}")
        return f"An error occurred: {type(e).__name__}"


def chat_with_llm(text, user_input, conversation_history, lang_code):
    """Starts a conversational interaction with the LLM, enforcing the specified language."""
    effective_lang_code = lang_code if lang_code else "en"
    print(f"LLM instructed to respond in: {effective_lang_code}")

    language_instruction = f"**IMPORTANT: You MUST respond ONLY in the language identified by the code: {effective_lang_code}.** Do not use any other language."
    prompt_parts = ["You are a helpful assistant."]

    if text:
        prompt_parts.extend([
            f"The language of the video transcript is '{effective_lang_code}'.",
            "Summarize the video transcript using bulet points when fits context and answer the user's question. If user asks follow up question and the answer in not found in transcript, then use your own knowlege to answer the follow up question.",
            language_instruction,
            "\n--- Video Transcript ---",
            text,
            "--- End Transcript ---"
        ])
    else:
        prompt_parts.extend([
            language_instruction,
            "Answer the user's question based on the chat history."
        ])

    prompt_parts.extend([
        "\n--- Chat History ---",
        conversation_history,
        "--- End History ---",
        f"\nUser: {user_input}",
        f"\nFormat your response in Markdown.",
        f"LLM ({effective_lang_code}):"
    ])

    prompt = "\n".join(prompt_parts)
    return call_gemini_api(prompt)

def summarize_text_with_llm(text_to_summarize, prompt_instruction):
    """
    Summarizes provided text using the Gemini LLM based on a given instruction.
    """
    full_prompt = f"""
    {prompt_instruction}

    Text to summarize:
    {text_to_summarize}
    """
    return call_gemini_api(full_prompt)