import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from search.utils import build_result

class DenseTransformerSearch:
    def __init__(self, data_path='data/processed_movies.csv', embeddings_path='data/sentence_embeddings.npy'):
        """
        Carrega a base de dados mapeada e os embeddings pré-computados via modelos de Transformers.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = pd.read_csv(data_path)
        self.embeddings = np.load(embeddings_path)
        
    def search(self, query, top_k=5):
        """
        Codifica a query em tempo real e computa a similaridade exata de cosseno de forma linear O(N).
        Retorna uma tupla: (lista_de_dicts padronizados, tempo gasto).
        """
        start_time = time.time()
        query_vector = self.model.encode([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            score = similarities[idx]
            results.append(build_result(row, score))
            
        search_time = time.time() - start_time
        
        return results, search_time

if __name__ == "__main__":
    # Teste de execução rápida do Sentence Transformers
    searcher = DenseTransformerSearch()
    results, duration = searcher.search("romantic comedy in New York")
    
    import json
    print(f"Tempo de busca: {duration:.4f}s\n")
    print("Top 2 Resultados (Sentence Transformers):")
    print(json.dumps(results[:2], indent=2, ensure_ascii=False))