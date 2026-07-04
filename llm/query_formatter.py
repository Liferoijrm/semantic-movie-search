import re
import requests

def clean_text_syntax(text):
    """
    Remove pontuações e caracteres especiais da consulta para padronizar a entrada.
    """
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'[^\w\s]', '', text)
    return cleaned.lower().strip()

def format_query(raw_query):
    """
    Detecta o idioma da consulta: se for inglês, apenas formata a string sintaticamente;
    se for português, aciona o LLM para tradução antes da higienização básica.
    """
    if not raw_query.strip():
        return ""

    lower_query = raw_query.lower()
    
    english_indicators = {'the', 'of', 'and', 'to', 'a', 'is', 'in', 'that', 'it', 'with', 'about', 'movie', 'film'}
    
    query_words = set(lower_query.split())
    
    if query_words.intersection(english_indicators):
        return clean_text_syntax(raw_query)

    ollama_url = "http://localhost:11434/api/generate"
    
    translation_prompt = f"""Translate the following movie search query from Portuguese to English. 
Return ONLY the direct English translation. Do not include quotes, explanations, or introductory text.

Query: "{raw_query}"
Translation:"""

    payload = {
        "model": "tinyllama",
        "prompt": translation_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=10)
        if response.status_code == 200:
            translated_query = response.json().get('response', raw_query).strip()
            return clean_text_syntax(translated_query)
    except requests.exceptions.RequestException:
        pass
        
    return clean_text_syntax(raw_query)