import os
from dotenv import load_dotenv
import google.generativeai as genai
def call_gemini(prompt, stream=True, model_name='gemini-1.5-pro'):
    """
    Calls Gemini LLM with the given prompt. Supports streaming output.
    Args:
        prompt (str): The input prompt for Gemini.
        stream (bool): Whether to stream the response.
        model_name (str): Gemini model to use.
    Returns:
        If stream=True: yields text chunks as they arrive.
        If stream=False: returns the full response text.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    if stream:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            yield chunk.text
    else:
        response = model.generate_content(prompt)
        return response.text
def get_truncated_gemini_answer(chunk_generator, max_words=40):
    """
    Consumes a Gemini streaming generator and returns the answer truncated to max_words words.
    """
    words = []
    for chunk in chunk_generator:
        for word in chunk.split():
            words.append(word)
            if len(words) >= max_words:
                return ' '.join(words) + '...'
    return ' '.join(words)
