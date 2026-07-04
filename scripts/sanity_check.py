import os
from sentence_transformers import SentenceTransformer

def check_dependencies():
    """
    Garante que o pipeline de dados foi rodado por completo antes de testar a busca.
    """
    required_files = [
        'data/sentence_embeddings.npy',
        'data/word2vec.model',
        'data/word2vec_embeddings.npy',
        'data/hnsw_index.faiss',
        'data/processed_movies.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        raise FileNotFoundError("Required data files are missing.")

def run_sanity_check():
    """
    Carrega as classes de busca uma vez e testa contra queries fixas de validação.
    """
    check_dependencies()
    
    from search.cosine_similarity import ClassicCosineSearch
    from search.hnsw_search import HNSWApproximateSearch
    from search.sentence_embeddings import DenseTransformerSearch
    from search.word2vec_average import Word2VecAverageSearch

    shared_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    searchers = {
        "ClassicCosineSearch": ClassicCosineSearch(),
        "Word2VecAverageSearch": Word2VecAverageSearch(),
        "DenseTransformerSearch": DenseTransformerSearch(model=shared_transformer),
        "HNSWApproximateSearch": HNSWApproximateSearch(model=shared_transformer)
    }

    queries = [
        "A hacker discovers the world is a simulation",
        "Philosophical reflections on the nature of time and space"
    ]

    for query in queries:
        for method_name, searcher in searchers.items():
            searcher.search(query, top_k=5)

if __name__ == "__main__":
    run_sanity_check()