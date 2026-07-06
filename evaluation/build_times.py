"""
Mede o tempo da fase OFFLINE (indexação/treinamento) de cada método de busca.

Diferente de `benchmark.py` (que mede o tempo de BUSCA, por query), aqui medimos o
custo pago UMA vez para construir cada estrutura a partir dos dados já limpos:

  - TF-IDF                -> ajuste (fit) da matriz TF-IDF
  - Word2Vec              -> treino do modelo + geração dos vetores-médio por filme
  - Sentence Embeddings   -> geração dos 42k embeddings com o MiniLM
                             (etapa compartilhada pelos métodos denso-linear e HNSW)
  - HNSW                  -> construção do índice FAISS sobre os embeddings

O script reproduz as operações dos scripts offline (`generate_initial_embeddings.py`
e `generate_resources.py`) apenas para cronometrá-las — não regrava nenhum artefato.
"""

from time import perf_counter

import numpy as np
import pandas as pd

DATA = "data/processed_movies.csv"
VECTOR_SIZE = 100  # dimensão do Word2Vec (igual ao generate_resources.py)


def timed(label, fn):
    """Executa `fn`, imprime e devolve (resultado, tempo em segundos)."""
    t0 = perf_counter()
    result = fn()
    dt = perf_counter() - t0
    print(f"  {label}: {dt:.2f}s")
    return result, dt


def measure_build_times():
    """
    Mede o tempo de construção de cada método e devolve (per_method, detail):
      - per_method: dict {nome_do_método -> segundos}, usando os MESMOS nomes de
        `benchmark.py` (facilita juntar as tabelas);
      - detail:     dict {etapa -> segundos} com o detalhamento por etapa.
    """
    df = pd.read_csv(DATA)
    summaries = df["Clean_summary"].astype(str)
    sentences = [s.split() for s in summaries]
    print(f"Documentos: {len(df)}\n")

    t = {}

    print("TF-IDF...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    _, t["tfidf_fit"] = timed(
        "fit_transform",
        lambda: TfidfVectorizer(stop_words="english").fit_transform(summaries),
    )

    print("Word2Vec...")
    from gensim.models import Word2Vec

    w2v, t["word2vec_train"] = timed(
        "treino",
        lambda: Word2Vec(
            sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4
        ),
    )

    def avg_vectors():
        def vec(words):
            vs = [w2v.wv[w] for w in words if w in w2v.wv]
            return np.mean(vs, axis=0) if vs else np.zeros(VECTOR_SIZE)

        return np.array([vec(s) for s in sentences])

    _, t["word2vec_vectors"] = timed("vetores-medio", avg_vectors)

    print("Sentence Embeddings (MiniLM)... (pode levar alguns minutos)")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb, t["embeddings_generation"] = timed(
        "encode dos documentos",
        lambda: model.encode(summaries.tolist(), show_progress_bar=False).astype(
            "float32"
        ),
    )

    print("HNSW (FAISS)...")
    import faiss

    def build_hnsw():
        e = emb.copy()
        faiss.normalize_L2(e)
        index = faiss.IndexHNSWFlat(e.shape[1], 32)
        index.add(e)
        return index

    _, t["hnsw_build"] = timed("build do indice", build_hnsw)

    # Composição por método: HNSW e Sentence-linear compartilham a geração dos
    # embeddings; Word2Vec = treino + vetores-medio.
    per_method = {
        "TF-IDF (linear)": t["tfidf_fit"],
        "Word2Vec (linear)": t["word2vec_train"] + t["word2vec_vectors"],
        "Sentence Emb. (linear)": t["embeddings_generation"],
        "Sentence Emb. (HNSW)": t["embeddings_generation"] + t["hnsw_build"],
    }
    return per_method, t


if __name__ == "__main__":
    per_method, detail = measure_build_times()

    print("\n" + "=" * 56)
    print("TEMPO DE CONSTRUCAO POR METODO (fase offline, uma vez)")
    print("=" * 56)
    for name, secs in per_method.items():
        print(f"  {name:24} {secs:8.2f}s")

    print("\nDetalhe das etapas:")
    for etapa, secs in detail.items():
        print(f"  {etapa:24} {secs:8.2f}s")
