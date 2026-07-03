import pandas as pd
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer
from search.utils import build_result

class HNSWApproximateSearch:
    def __init__(self, data_path='data/processed_movies.csv'):
        """
        Carrega o grafo HNSW pré-computado do disco de forma instantânea.
        """
        self.df = pd.read_csv(data_path)
        self.index = faiss.read_index('data/hnsw_index.faiss')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def search(self, query, top_k=5):
        """
        Navega pelo grafo HNSW de forma sublinear para encontrar os vizinhos mais próximos da query.
        Retorna uma tupla: (lista_de_dicts padronizados, tempo gasto).
        """
        start_time = time.time()
        query_vector = self.model.encode([query], normalize_embeddings=True).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            # FAISS retorna -1 no índice caso não encontre vizinhos suficientes no grafo
            if idx == -1:
                continue
                
            row = self.df.iloc[idx]
            
            # Conversão matemática de Distância L2^2 para Score de Similaridade (maior = melhor)
            l2_squared_distance = distances[0][i]
            score = 1.0 - (l2_squared_distance / 2.0)
            
            results.append(build_result(row, score))
            
        search_time = time.time() - start_time
        
        return results, search_time

if __name__ == "__main__":
    # Teste de execução do HNSW
    searcher = HNSWApproximateSearch()
    results, duration = searcher.search("romantic comedy in New York")
    
    import json
    print(f"Tempo de busca: {duration:.4f}s\n")
    print("Top 2 Resultados (HNSW FAISS):")
    print(json.dumps(results[:2], indent=2, ensure_ascii=False))