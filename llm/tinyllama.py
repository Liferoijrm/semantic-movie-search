import requests

def generate_response(user_query, retrieved_movies):
    """
    Monta o prompt de forma determinística e extrai do TinyLlama apenas 
    as tags solicitadas, pulando linhas entre os filmes e mitigando alucinações.
    """
    ollama_url = "http://localhost:11434/api/generate"
    
    context_block = ""
    for idx, movie in enumerate(retrieved_movies, 1):
        genres_str = ", ".join(movie['genres']) if movie['genres'] else "Unknown"
        context_block += f"### Movie Data:\nTitle: {movie['title']}\nGenre: {genres_str}\nSynopsis: {movie['summary'][:500]}...\n\n"

    # Criamos um prompt no formato de "Completion" direto, sem margem para conversas
    prompt = f"""You are a data formatting script. Your only job is to extract and format the matching movies from the context into the exact text template provided below. Do not say hello, do not write explanations, do not create introductions, and do not add any conversation.

Context:
{context_block}

User Request: "{user_query}"

Format each matching movie from the context strictly using this text structure, followed by a double line break:
- **Name**: [Insert Title here]
- **Genre**: [Insert Genre here]
- **Synopsis**: [Insert brief 1-sentence description based on the context]

Output:"""

    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,     # Temperatura zero mata a aleatoriedade e as alucinações
            "top_p": 0.1,
            "stop": ["###", "User:"] # Força a parada caso o modelo tente inventar dados adicionais
        }
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get('response', 'Error: Empty response from LLM.').strip()
        return f"_(Error: Local Ollama server returned status code {response.status_code})_"
        
    except requests.exceptions.RequestException:
        return "_(LLM Connection Error: Make sure Ollama is running local with 'ollama run tinyllama')_"