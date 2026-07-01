import pandas as pd
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer

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
        """
        start_time = time.time()
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        search_time = time.time() - start_time
        
        return self.df.iloc[indices[0]], search_time