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

    # STRONG language enforcement - system-level instructions
    system_instructions = {
        'en': "You are a helpful assistant that responds EXCLUSIVELY in English. NEVER use any other language. Your responses must be 100% in English.",
        'ru': "Вы - полезный помощник, который отвечает ИСКЛЮЧИТЕЛЬНО на русском языке. НИКОГДА не используйте другие языки. Ваши ответы должны быть на 100% на русском языке.",
        'fr': "Vous êtes un assistant utile qui répond EXCLUSIVEMENT en français. N'UTILISEZ JAMAIS d'autres langues. Vos réponses doivent être à 100% en français."
    }

    # Language-specific prompts
    summary_instructions = {
        'en': "Summarize the video transcript using bullet points when appropriate and answer the user's question. If user asks follow up question and the answer is not found in transcript, then use your own knowledge to answer the follow up question.",
        'ru': "Обобщите транскрипт видео, используя маркированные списки там, где это уместно, и ответьте на вопрос пользователя. Если пользователь задает уточняющий вопрос и ответ не найден в транскрипте, используйте свои знания, чтобы ответить на уточняющий вопрос.",
        'fr': "Résumez la transcription de la vidéo en utilisant des puces lorsque cela est approprié et répondez à la question de l'utilisateur. Si l'utilisateur pose une question de suivi et que la réponse ne se trouve pas dans la transcription, utilisez vos propres connaissances pour répondre à la question de suivi."
    }

    system_instruction = system_instructions.get(effective_lang_code,
                                                 f"You are a helpful assistant that responds EXCLUSIVELY in {effective_lang_code}. NEVER use any other language.")

    summary_instruction = summary_instructions.get(effective_lang_code,
                                                   "Summarize the video transcript using bullet points when appropriate and answer the user's question. If user asks follow up question and the answer is not found in transcript, then use your own knowledge to answer the follow up question.")

    # STRONG final instruction
    final_instructions = {
        'en': "**CRITICAL: RESPOND ONLY IN ENGLISH. DO NOT USE ANY OTHER LANGUAGE.**",
        'ru': "**КРИТИЧЕСКИ ВАЖНО: ОТВЕЧАЙТЕ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. НЕ ИСПОЛЬЗУЙТЕ ДРУГИЕ ЯЗЫКИ.**",
        'fr': "**CRITIQUE: RÉPONDEZ UNIQUEMENT EN FRANÇAIS. N'UTILISEZ AUCUNE AUTRE LANGUE.**"
    }

    final_instruction = final_instructions.get(effective_lang_code,
                                               f"**CRITICAL: RESPOND ONLY IN {effective_lang_code.upper()}. DO NOT USE ANY OTHER LANGUAGE.**")

    prompt_parts = [system_instruction]

    if text:
        prompt_parts.extend([
            f"Video transcript language: {effective_lang_code}",
            summary_instruction,
            "\n--- Video Transcript ---",
            text,
            "--- End Transcript ---"
        ])
    else:
        prompt_parts.extend([
            summary_instruction
        ])

    prompt_parts.extend([
        "\n--- Chat History ---",
        conversation_history,
        "--- End History ---",
        f"\nUser: {user_input}",
        final_instruction,
        f"Format your response in Markdown.",
        f"LLM ({effective_lang_code}):"
    ])

    prompt = "\n".join(prompt_parts)
    return call_gemini_api(prompt)


def translate_text_with_llm(text, target_language):
    """
    Translates text to the specified target language using the LLM.
    """
    if target_language == 'ru':
        translation_prompt = f"""
        **КРИТИЧЕСКИ ВАЖНО: Вы ДОЛЖНЫ перевести следующий текст НА РУССКИЙ ЯЗЫК.**
        **НИКОГДА не оставляйте текст на английском или других языках.**
        **Перевод должен быть полным и точным.**

        Текст для перевода:
        {text}

        Переведенный текст на русский язык:
        """
    elif target_language == 'fr':
        translation_prompt = f"""
        **CRITIQUE: Vous DEVEZ traduire le texte suivant EN FRANÇAIS.**
        **NE LAISSEZ JAMAIS le texte en anglais ou dans d'autres langues.**
        **La traduction doit être complète et précise.**

        Texte à traduire:
        {text}

        Texte traduit en français:
        """
    else:
        translation_prompt = f"""
        **CRITICAL: You MUST translate the following text TO {target_language.upper()}.**
        **NEVER leave any text in English or other languages.**
        **The translation must be complete and accurate.**

        Text to translate:
        {text}

        Translated text in {target_language}:
        """

    return call_gemini_api(translation_prompt)


def detect_hallucination(response_text, original_transcript):
    """
    Detects if the LLM is hallucinating by checking for common generic patterns.
    Returns True if hallucination is detected, False otherwise.
    """
    if not response_text or not original_transcript:
        return False

    response_lower = response_text.lower()

    # Common hallucination patterns
    hallucination_indicators = [
        "i cannot directly access",
        "i cannot play video content",
        "hypothetical video transcript",
        "generic summary",
        "common informative topic",
        "if the video transcript were discussing",
        "this is a hypothetical",
        "i don't have access to",
        "unable to access the video",
        "cannot view external links"
    ]

    # Check for any hallucination indicators
    for indicator in hallucination_indicators:
        if indicator in response_lower:
            print(f"DEBUG: Hallucination detected - indicator: {indicator}")
            return True

    # Check if response mentions time management (common hallucination topic)
    time_management_terms = [
        "time management", "effective time", "productivity", "eisenhower matrix",
        "smart goals", "pomodoro technique", "eat the frog"
    ]

    # If response contains time management terms but transcript doesn't, likely hallucination
    if any(term in response_lower for term in time_management_terms):
        transcript_lower = original_transcript.lower()
        if not any(term in transcript_lower for term in time_management_terms):
            print("DEBUG: Hallucination detected - time management terms without context")
            return True

    return False


def summarize_text_with_llm(text_to_summarize, prompt_instruction, lang_code='en'):
    """
    Summarizes provided text using the Gemini LLM based on a given instruction.
    Includes hallucination detection and retry logic.
    """
    max_retries = 2

    for attempt in range(max_retries + 1):
        print(f"DEBUG: Summarization attempt {attempt + 1}/{max_retries + 1}")

        # Language-specific summary instructions with explicit transcript requirements
        summary_instructions = {
            'en': f"""
{prompt_instruction}

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
1. **USE ONLY THE PROVIDED TRANSCRIPT BELOW** - Do NOT generate hypothetical or generic content
2. **RESPOND ONLY IN ENGLISH** - Do not use any other language
3. **DO NOT HALLUCINATE** - Only summarize what's actually in the transcript
4. **IGNORE ANY EXTERNAL LINKS** - We are providing the transcript directly below
5. **DO NOT SAY YOU CANNOT ACCESS THE VIDEO** - The transcript is provided below
6. **THIS IS NOT A HYPOTHETICAL** - Use the actual transcript provided

**VIDEO TRANSCRIPT (USE THIS EXACT CONTENT):**
{text_to_summarize}

**FINAL REMINDER: Summarize ONLY the transcript provided above. Do NOT make up content.**
""",
            'ru': f"""
{prompt_instruction}

**КРИТИЧЕСКИ ВАЖНЫЕ ИНСТРАКЦИИ - ВНИМАТЕЛЬНО ПРОЧИТАЙТЕ:**
1. **ИСПОЛЬЗУЙТЕ ТОЛЬКО ПРЕДОСТАВЛЕННЫЙ ТРАНСКРИПТ НИЖЕ** - Не создавайте гипотетический или общий контент
2. **ОТВЕЧАЙТЕ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ** - Не используйте другие языки
3. **НЕ ВЫДУМЫВАЙТЕ** - Обобщайте только то, что действительно есть в транскрипте
4. **ИГНОРИРУЙТЕ ВНЕШНИЕ ССЫЛКИ** - Мы предоставляем транскрипт напрямую ниже
5. **НЕ ГОВОРИТЕ, ЧТО НЕ МОЖЕТЕ ПОЛУЧИТЬ ДОСТУП К ВИДЕО** - Транскрипт предоставлен ниже
6. **ЭТО НЕ ГИПОТЕТИЧЕСКИЙ СЦЕНАРИЙ** - Используйте предоставленный транскрипт

**ТРАНСКРИПТ ВИДЕО (ИСПОЛЬЗУЙТЕ ЭТОТ КОНКРЕТНЫЙ КОНТЕНТ):**
{text_to_summarize}

**ФИНАЛЬНОЕ НАПОМИНАНИЕ: Обобщите ТОЛЬКО предоставленный выше транскрипт. НЕ выдумывайте контент.**
""",
            'fr': f"""
{prompt_instruction}

**INSTRUCTIONS CRITIQUES - LISEZ ATTENTIVEMENT:**
1. **UTILISEZ UNIQUEMENT LA TRANSCRIPTION FOURNIE CI-DESSOUS** - Ne générez pas de contenu hypothétique ou générique
2. **RÉPONDEZ UNIQUEMENT EN FRANÇAIS** - N'utilisez aucune autre langue
3. **NE HALLUCINEZ PAS** - Résumez uniquement ce qui se trouve réellement dans la transcription
4. **IGNOREZ TOUS LES LIENS EXTERNES** - Nous fournissons la transcription directement ci-dessous
5. **NE DITES PAS QUE VOUS NE POUVEZ PAS ACCÉDER À LA VIDÉO** - La transcription est fournie ci-dessous
6. **CE N'EST PAS UN SCÉNARIO HYPOTHÉTIQUE** - Utilisez la transcription fournie

**TRANSCRIPTION VIDÉO (UTILISEZ CE CONTENU EXACT):**
{text_to_summarize}

**RAPPEL FINAL: Résumez UNIQUEMENT la transcription fournie ci-dessus. N'INVENTEZ PAS de contenu.**
"""
        }

        summary_prompt = summary_instructions.get(lang_code, f"""
{prompt_instruction}

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
1. **USE ONLY THE PROVIDED TRANSCRIPT BELOW** - Do NOT generate hypothetical or generic content
2. **RESPOND ONLY IN {lang_code.upper()}** - Do not use any other language
3. **DO NOT HALLUCINATE** - Only summarize what's actually in the transcript
4. **IGNORE ANY EXTERNAL LINKS** - We are providing the transcript directly below
5. **DO NOT SAY YOU CANNOT ACCESS THE VIDEO** - The transcript is provided below
6. **THIS IS NOT A HYPOTHETICAL** - Use the actual transcript provided

**VIDEO TRANSCRIPT (USE THIS EXACT CONTENT):**
{text_to_summarize}

**FINAL REMINDER: Summarize ONLY the transcript provided above. Do NOT make up content.**
""")

        response = call_gemini_api(summary_prompt)

        # Check for hallucination
        if not detect_hallucination(response, text_to_summarize):
            print("DEBUG: Summary appears to be based on actual transcript")
            return response
        else:
            print(f"DEBUG: Hallucination detected in attempt {attempt + 1}")
            if attempt < max_retries:
                print("DEBUG: Retrying with stronger instructions...")
                # Add stronger instructions for retry
                if attempt == 0:
                    summary_prompt += "\n\n**SECOND ATTEMPT - YOU MUST USE THE PROVIDED TRANSCRIPT. DO NOT IGNORE IT.**"
                else:
                    summary_prompt += "\n\n**FINAL ATTEMPT - YOU ARE REQUIRED TO USE THE TRANSCRIPT PROVIDED ABOVE. FAILURE TO DO SO WILL RESULT IN AN ERROR.**"
            else:
                print("DEBUG: Max retries reached, returning error message")
                error_msg = {
                    'en': "The AI was unable to generate a summary based on the provided transcript. The response contained generic content instead of analyzing the actual video content.",
                    'ru': "ИИ не смог сгенерировать сводку на основе предоставленного транскрипта. В ответе содержался общий контент вместо анализа фактического содержания видео.",
                    'fr': "L'IA n'a pas pu générer un résumé basé sur la transcription fournie. La réponse contenait du contenu générique au lieu d'analyser le contenu réel de la vidéo."
                }
                return error_msg.get(lang_code, "Unable to generate summary based on the provided transcript.")

    return "Unable to generate summary based on the provided transcript."
