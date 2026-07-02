import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        """
        start_time = time.time()
        query_vector = self.model.encode([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        search_time = time.time() - start_time
        
        results = []
        for idx in top_indices:
            movie_data = self.df.iloc[idx]
            results.append({
                'title': movie_data['Movie_name'],
                'genres': movie_data['Movie_genres'],
                'similarity_score': similarities[idx],
                'summary': movie_data['Clean_summary']
            })
            
        return results, search_time

if __name__ == "__main__":
    searcher = DenseTransformerSearch()
    results, duration = searcher.search("romantic comedy in New York")