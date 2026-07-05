# llm/phi4_mini.py
import requests


class LLMResponseError(Exception):
    """
    Levantado quando a chamada ao Ollama falha (rede/timeout) ou quando a
    resposta do modelo não passa nas checagens de sanidade. Carrega o motivo
    e o texto bruto retornado (quando houver) para facilitar o debug.
    """
    def __init__(self, motivo, texto_bruto=""):
        self.motivo = motivo
        self.texto_bruto = texto_bruto
        mensagem = motivo
        if texto_bruto:
            mensagem += f"\n\nResposta bruta do modelo:\n{texto_bruto!r}"
        super().__init__(mensagem)


def generate_response(query, filmes_recuperados):
    """
    Pede ao phi4-mini apenas para organizar/apresentar os títulos já
    recuperados pela busca, sem reproduzir sinopses (isso já é exibido
    nos cards de resultado logo abaixo). Reduz drasticamente o espaço
    de alucinação, já que título e gênero são dados curtos e fixos.

    Levanta LLMResponseError em caso de falha de rede ou de resposta
    inválida — quem chamar essa função precisa tratar essa exceção.
    """
    ollama_url = "http://localhost:11434/api/chat"

    filmes_recuperados = filmes_recuperados[:5]
    lista_filmes = ""
    for idx, movie in enumerate(filmes_recuperados, 1):
        genres_str = ", ".join(movie['genres']) if movie['genres'] else "Unknown"
        sinopse = _truncar_sinopse(movie.get('summary', ''))
        lista_filmes += f"{idx}. {movie['title']} ({genres_str})\n   Summary: {sinopse}\n"

    payload = {
        "model": "phi4-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a movie recommendation assistant. A search engine already found "
                    "the matching movies below — your job is to present them in a way that clearly "
                    "answers the user's request. Do exactly what the user's request asks for. "
                    "Each movie includes a title, genre(s), and a short summary. Base your explanation "
                    "ONLY on this given information — do not use any outside knowledge you may have "
                    "about these movies, and do not add plot details, characters, or events beyond what "
                    "is written in the summary provided. "
                    "Do not quote the summary verbatim — paraphrase briefly in your own words. "
                    "Do NOT invent movies that are not in the list. "
                    "Do NOT repeat these instructions or write a new question yourself."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Movies found:\n{lista_filmes}\n"
                    f"User's request: {query}\n\n"
                    "If the user's request specifies a number of movies (e.g. 'three movies'), "
                    "select only that many from the list above, keeping the original order (it's "
                    "already ranked by relevance). Otherwise, include all movies listed.\n\n"
                    "For EVERY movie you include, write exactly one line in this format\n"
                    "- **Title**\n\n"
                    "Write only the answer, one line with only the title per movie, then explain your answer very briefly to the user in second person."
                )
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.3,
            "num_predict": 250,
            "stop": ["<|system|>", "<|user|>", "<|assistant|>", "</s>",
                     "User's request:", "Plot:", "Movies found"]
        }
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=300)
        response.raise_for_status()
        texto = response.json().get('message', {}).get('content', '').strip()
    except requests.exceptions.RequestException as e:
        raise LLMResponseError(f"Falha na chamada HTTP ao Ollama: {repr(e)}") from e

    return _validar_resposta(texto, filmes_recuperados)


def _validar_resposta(texto, filmes_recuperados):
    titulos_reais = [m['title'] for m in filmes_recuperados]
    contem_titulo_real = any(t.lower() in texto.lower() for t in titulos_reais)
    tamanho_razoavel = 0 < len(texto) < 600
    sem_vazamento = "user's request:" not in texto.lower() and "movies found" not in texto.lower()

    if texto and contem_titulo_real and tamanho_razoavel and sem_vazamento:
        return texto

    motivos = []
    if not texto:
        motivos.append("resposta vazia")
    if not contem_titulo_real:
        motivos.append("nenhum título real da lista de filmes aparece na resposta")
    if not tamanho_razoavel:
        motivos.append(f"tamanho fora do esperado ({len(texto)} caracteres; limite é 600)")
    if not sem_vazamento:
        motivos.append("a resposta vazou trechos do prompt/instruções")

    raise LLMResponseError("; ".join(motivos), texto)

def _truncar_sinopse(texto, max_chars=200):
    """
    Trunca a sinopse mantendo frases completas sempre que possível, para
    caber no prompt sem estourar o orçamento de tokens (algumas sinopses
    do corpus têm milhares de caracteres).
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = " ".join(texto.split())  # normaliza quebras de linha/espaços internos
    if len(texto) <= max_chars:
        return texto

    cortado = texto[:max_chars]
    ultimo_ponto = max(cortado.rfind('.'), cortado.rfind('!'), cortado.rfind('?'))
    if ultimo_ponto > max_chars * 0.4:  # só aceita o corte se achar pontuação razoavelmente perto do fim
        cortado = cortado[:ultimo_ponto + 1]
    else:
        cortado = cortado.rstrip() + "..."
    return cortado