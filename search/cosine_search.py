import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

class SemanticSearch:
    def __init__(self, df_path='data/processed_movies.csv', embeddings_path='data/sentence_embeddings.npy'):
        print("Carregando modelo de linguagem (Sentence Transformers)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Carregando base de dados de filmes...")
        self.df = pd.read_csv(df_path)
        
        print("Carregando embeddings dos filmes (matriz de vetores)...")
        self.embeddings = np.load(embeddings_path)
        
    def search(self, user_query, top_k=5):
        """
        Dada uma pergunta em linguagem natural, retorna os top_k filmes mais relevantes.
        """
        start_time = time.time()
        
        # 1. Converter a pergunta do usuário para um vetor (mesma dimensionalidade dos filmes)
        query_vector = self.model.encode([user_query])
        
        # 2. Calcular a Similaridade de Cosseno
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # 3. Pegar os índices dos filmes com a maior similaridade
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # 4. Formatar os resultados
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
    pass