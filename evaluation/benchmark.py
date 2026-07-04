"""
Benchmark comparativo dos quatro métodos de busca semântica.

Mede, por método:
  - Qualidade  -> hit@k e recall@k (vários k) e MRR contra as queries de avaliação
  - Inferência -> tempo médio de busca por query

Também reporta as métricas por categoria (iconic/obscure/multi/user), o que expõe
o viés de popularidade — a principal análise de qualidade do enunciado.

Cronometragem: time.perf_counter, com warm-up antes de medir (a 1ª chamada ao
encoder/índice é mais lenta) e o MENOR de REPEATS execuções por query (reduz ruído
de GC/escalonamento). Busca-se uma vez com top_k = max(K_VALUES) e os k menores
saem por fatiamento — exato nos métodos lineares; no HNSW pode diferir marginalmente.
"""

from time import perf_counter

from evaluation.queries import ALL_QUERIES
from evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank

K_VALUES = [1, 3, 5, 10]  # valores de k para hit@k e recall@k
TOP_K = max(K_VALUES)  # busca-se uma vez com o maior k e derivam-se os menores
REPEATS = 5  # repetições por query; mantém-se o menor tempo


def build_searchers():
    """Instancia os quatro buscadores, compartilhando um único MiniLM entre os densos."""
    from sentence_transformers import SentenceTransformer
    from search.cosine_similarity import ClassicCosineSearch
    from search.word2vec_average import Word2VecAverageSearch
    from search.sentence_embeddings import DenseTransformerSearch
    from search.hnsw_search import HNSWApproximateSearch

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return {
        "TF-IDF (linear)": ClassicCosineSearch(),
        "Word2Vec (linear)": Word2VecAverageSearch(),
        "Sentence Emb. (linear)": DenseTransformerSearch(model=model),
        "Sentence Emb. (HNSW)": HNSWApproximateSearch(model=model),
    }


def collect_raw(searchers, queries=ALL_QUERIES, top_k=TOP_K, repeats=REPEATS):
    """
    Roda cada método sobre todas as queries e devolve os dados crus.

    Retorna, por método, duas listas alinhadas à ordem de `queries`:
      - "times":     menor tempo de busca (s) em `repeats` execuções;
      - "retrieved": ids recuperados (top_k, em ordem de ranking).
    Guardar os ids crus (em vez de já calcular métricas) permite derivar qualquer
    métrica/qualquer k depois.
    """
    for item in queries:
        assert item["query"].strip(), f"Query vazia no gabarito: {item['titles']}"

    # Warm-up: aquece encoder/índice para não contaminar a 1ª medição.
    for searcher in searchers.values():
        searcher.search("a warm up query to trigger lazy initialization", top_k=top_k)

    raw = {name: {"times": [], "retrieved": []} for name in searchers}
    for item in queries:
        for name, searcher in searchers.items():
            best_time = float("inf")
            for _ in range(repeats):
                t0 = perf_counter()
                results, _internal = searcher.search(item["query"], top_k=top_k)
                best_time = min(best_time, perf_counter() - t0)
            raw[name]["times"].append(best_time)
            raw[name]["retrieved"].append([int(r["movie_id"]) for r in results])
    return raw


def _aggregate(raw_method, queries, indices, k_values):
    """Média das métricas de um método sobre um subconjunto de queries (por índice)."""
    n = len(indices)
    if n == 0:
        return None

    agg = {"n": n, "search_ms": 0.0, "mrr": 0.0}
    for k in k_values:
        agg[f"hit@{k}"] = agg[f"recall@{k}"] = 0.0

    for i in indices:
        retrieved = raw_method["retrieved"][i]
        expected = queries[i]["expected_ids"]
        agg["search_ms"] += raw_method["times"][i] * 1000
        agg["mrr"] += reciprocal_rank(retrieved, expected)
        for k in k_values:
            agg[f"hit@{k}"] += hit_at_k(retrieved, expected, k)
            agg[f"recall@{k}"] += recall_at_k(retrieved, expected, k)

    for key in list(agg):
        if key != "n":
            agg[key] /= n
    return agg


def aggregate(raw, queries=ALL_QUERIES, k_values=K_VALUES):
    """Métricas sobre TODAS as queries, uma linha por método."""
    indices = list(range(len(queries)))
    rows = []
    for name, data in raw.items():
        agg = _aggregate(data, queries, indices, k_values)
        agg["method"] = name
        rows.append(agg)
    return rows


def aggregate_by_category(raw, queries=ALL_QUERIES, k_values=K_VALUES):
    """Métricas por (método, categoria) — evidencia o viés de popularidade."""
    categories = list(dict.fromkeys(q["category"] for q in queries))
    index_by_cat = {
        c: [i for i, q in enumerate(queries) if q["category"] == c] for c in categories
    }

    rows = []
    for name, data in raw.items():
        for cat in categories:
            agg = _aggregate(data, queries, index_by_cat[cat], k_values)
            agg["method"] = name
            agg["category"] = cat
            rows.append(agg)
    return rows


def _print_table(header, rows):
    """Imprime uma tabela de texto com colunas alinhadas (larguras automáticas)."""
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    print(line(header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(line(r))


def print_results(overall, by_cat, k_values=K_VALUES):
    """Imprime as duas tabelas (geral e por categoria) na tela."""
    hit_cols = [f"hit@{k}" for k in k_values]

    print(f"\nGERAL — {overall[0]['n']} queries")
    _print_table(
        ["Método", "Busca(ms)"] + hit_cols + ["MRR"],
        [
            [r["method"], f"{r['search_ms']:.1f}"]
            + [f"{r[c]:.2f}" for c in hit_cols]
            + [f"{r['mrr']:.3f}"]
            for r in overall
        ],
    )

    k = 5 if 5 in k_values else k_values[-1]
    print(f"\nPOR CATEGORIA (k={k})")
    _print_table(
        ["Método", "Categoria", "n", f"hit@{k}", f"recall@{k}", "MRR"],
        [
            [
                r["method"],
                r["category"],
                r["n"],
                f"{r[f'hit@{k}']:.2f}",
                f"{r[f'recall@{k}']:.2f}",
                f"{r['mrr']:.3f}",
            ]
            for r in by_cat
        ],
    )


if __name__ == "__main__":
    print("Carregando buscadores (pode levar alguns segundos)...")
    searchers = build_searchers()
    print("Rodando benchmark...")
    raw = collect_raw(searchers)
    print_results(aggregate(raw), aggregate_by_category(raw))
