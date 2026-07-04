import streamlit as st
import time

try:
    from llm.query_formatter import format_query
except ImportError:
    def format_query(query_raw):
        return query_raw

try:
    from llm.tinyllama import generate_response
except ImportError:
    def generate_response(query, retrieved_movies):
        return "_(LLM ainda não integrado — mostrando apenas resultados brutos abaixo)_"

from search.cosine_similarity import ClassicCosineSearch
from search.hnsw_search import HNSWApproximateSearch
from search.sentence_embeddings import DenseTransformerSearch
from search.word2vec_average import Word2VecAverageSearch
from sentence_transformers import SentenceTransformer
from scripts.sanity_check import check_dependencies

st.set_page_config(
    page_title="Semantic Movie Search",
    page_icon="assets/icon.png",
    layout="wide"
)

@st.cache_resource(show_spinner="Carregando modelos e índices (Isso pode levar alguns segundos)...")
def load_searchers():
    """
    Checa dependências, carrega o modelo pesado uma única vez e 
    distribui para as classes, retornando o dicionário pronto.
    """
    check_dependencies()
    shared_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    return {
        "TF-IDF": ClassicCosineSearch(),
        "Word2Vec": Word2VecAverageSearch(),
        "Sentence Embeddings": DenseTransformerSearch(model=shared_transformer),
        "HNSW": HNSWApproximateSearch(model=shared_transformer),
    }

try:
    searchers = load_searchers()
except FileNotFoundError:
    st.error("Faltam arquivos de dados — rode o pipeline de geração (`clean_data.py` e `generate_resources.py`) primeiro.")
    st.stop()

st.title("Semantic Movie Search")
st.markdown("""
Esta ferramenta permite buscar filmes no dataset da Wikipedia (CMU Movie Summary Corpus) usando **contexto e significado**, 
em vez de apenas palavras exatas. Explore as diferenças entre métodos clássicos de busca semântica e modelos neurais densos.
""")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    selected_method = st.radio(
        "Escolha o algoritmo de busca:",
        options=list(searchers.keys()) + ["Comparar todos (Ainda não funcional)"],
        horizontal=True
    )

with col2:
    top_k = st.slider("Resultados por método (top_k):", min_value=1, max_value=10, value=5)

st.write("") 

with st.form(key='search_form'):
    query = st.text_input("Digite o enredo, tema ou conceito do filme que você procura:", 
                          placeholder="Ex: A hacker discovers the world is a simulation...")
    submit_button = st.form_submit_button(label="Buscar Filmes")

if submit_button:
    if not query.strip():
        st.warning("Por favor, digite algum texto para realizar a busca (em inglês).")
    else:
        query_formatted = format_query(query)
        st.caption(f"Busca interpretada: _{query_formatted}_")

        if selected_method == "Comparar todos (Demo)":
            results, duration = searchers["Sentence Embeddings"].search(query_formatted, top_k=top_k)
        else:
            results, duration = searchers[selected_method].search(query_formatted, top_k=top_k)

        with st.chat_message("assistant"):
            with st.spinner("Formatando resposta..."):
                response_text = generate_response(query_formatted, results)
            st.markdown(response_text)

        st.divider()
        st.subheader(f"Resultados brutos recuperados ({duration:.3f}s)")

        if not results:
            st.info("Nenhum resultado encontrado.")
        else:
            for res in results:
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{res['title']}**")
                        genres_str = ", ".join(res['genres']) if res['genres'] else "Sem gênero"
                        st.caption(genres_str)
                    with c2:
                        st.metric("Score", f"{res['score']:.3f}")
                    with st.expander("Ver sinopse"):
                        st.write(res['summary'])