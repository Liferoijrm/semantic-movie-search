import os
import sys

def check_dependencies():
    """
    Garante que o pipeline de dados foi rodado por completo antes de testar a busca.
    Evita stack traces crus caso falte algum arquivo.
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
        print("❌ Erro: Os seguintes arquivos obrigatórios não foram encontrados:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nExecute o pipeline de geração (clean_data.py e generate_resources.py) primeiro.")
        # SUBSTITUA O sys.exit(1) PELA LINHA ABAIXO:
        raise FileNotFoundError("Arquivos obrigatórios ausentes.")
        
    print("✅ Todos os arquivos de dados necessários foram encontrados.\n")

def run_sanity_check():
    """
    Carrega as classes de busca uma vez e testa contra queries fixas.
    """
    check_dependencies()

    print("Carregando modelos, embeddings e índices (isso pode levar alguns segundos)...\n")
    
    # Importes absolutos assumindo execução a partir da raiz
    from search.cosine_similarity import ClassicCosineSearch
    from search.hnsw_search import HNSWApproximateSearch
    from search.sentence_embeddings import DenseTransformerSearch
    from search.word2vec_average import Word2VecAverageSearch

    # Instancia as 4 classes antes do loop para poupar processamento e memória
    searchers = {
        "ClassicCosineSearch": ClassicCosineSearch(),
        "Word2VecAverageSearch": Word2VecAverageSearch(),
        "DenseTransformerSearch": DenseTransformerSearch(),
        "HNSWApproximateSearch": HNSWApproximateSearch()
    }

    # Definição das queries cobrindo diferentes propósitos
    queries = [
        "A hacker discovers the world is a simulation",                # Específica / Literal
        "Philosophical reflections on the nature of time and space",   # Vaga / Temática
        "A radioactive platypus baking pizzas in medieval France"      # Difícil (sem match forte)
    ]

    for query in queries:
        print("\n" + "#" * 100)
        print(f"🔎 INICIANDO BATERIA PARA QUERY: '{query}'")
        print("#" * 100 + "\n")

        for method_name, searcher in searchers.items():
            results, duration = searcher.search(query, top_k=5)
            
            # Formatação exata solicitada no Passo 6
            print(f"=== {method_name} | query: \"{query}\" | tempo: {duration:.3f}s ===")
            
            if not results:
                print("  [Nenhum resultado retornado]")
            else:
                for pos, res in enumerate(results, 1):
                    # Garante que gêneros ausentes não quebrem o display
                    genres_str = ", ".join(res['genres']) if res['genres'] else "Sem gênero"
                    print(f"  {pos:02d}. {res['title']:<40} | Score: {res['score']:.3f} | Gêneros: [{genres_str}]")
            print() # Linha em branco extra para respiro visual

if __name__ == "__main__":
    run_sanity_check()