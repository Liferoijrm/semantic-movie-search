# llm/query_formatter.py
import re

# 1. Ruído de Comando (Remover para TODOS os métodos)
COMMAND_PREFIXES = [
    r'\btell me\b', r'\bgive me\b', r'\bshow me\b', r'\bfind me\b',
    r'\brecommend( me)?\b', r'\bsuggest( me)?\b',
    r'\bi want\b', r'\bi would like\b', r'\bi\'d like\b',
    r'\bcan you\b', r'\bcould you\b', r'\bplease\b', r'\blist\b',
]

COUNT_PHRASES = [
    r'\b(a|an|one|two|three|four|five|six|seven|\d+)\s+movies?\b',
    r'\b(some|a few|several|a couple( of)?)\s+movies?\b',
]

# 2. Conectivos de Domínio (Remover para TODOS, pois indicam a transição do comando pro enredo)
# Ex: "(tell me a movie) *about* a guy who..."
DOMAIN_CONNECTIVES = [
    r'\bmovies?\s+about\b', r'\bfilms?\s+about\b', r'\bstory\s+about\b',
    r'\bplot\s+about\b', r'\bmovies?\s+where\b', r'\bfilms?\s+where\b'
]

# 3. Stop Words Padrões (Remover APENAS para Word2Vec/TF-IDF)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "don't",
    "dont", "doesn't", "doesnt", 'title'
}

def clean_text_syntax(text):
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'[^\w\s]', '', text)
    return cleaned.lower().strip()

def strip_command_noise(text):
    cleaned = text
    for pattern in COMMAND_PREFIXES + COUNT_PHRASES + DOMAIN_CONNECTIVES:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned).strip()

def format_query_for_transformers(raw_query):
    """
    Uso: Sentence Embeddings e HNSW.
    Preserva a gramática e as stop words para manter o contexto sintático,
    removendo apenas os verbos de comando e quantificadores.
    """
    if not raw_query.strip():
        return ""
    # Retorna o texto limpo de comandos, mas com as preposições/artigos intactos
    return clean_text_syntax(strip_command_noise(raw_query))

def format_query_for_word2vec(raw_query):
    """
    Uso: Word2Vec Average e TF-IDF/Cosseno.
    Limpeza agressiva: remove comandos e também todas as stop words
    para evitar que vetores irrelevantes puxem a média/tfidf para baixo.
    """
    if not raw_query.strip():
        return ""
    
    # Primeiro tira o ruído de comando
    text_no_commands = strip_command_noise(raw_query)
    
    # Limpa pontuação e joga pra minúsculo
    normalized_text = clean_text_syntax(text_no_commands)
    
    # Remove as stop words
    words = normalized_text.split()
    meaningful_words = [word for word in words if word not in STOP_WORDS]
    
    return " ".join(meaningful_words)