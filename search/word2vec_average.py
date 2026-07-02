import pandas as pd
import numpy as np
import time
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecAverageSearch:
    def __init__(self, data_path='data/processed_movies.csv'):
        """
        Carrega o modelo Word2Vec e a matriz de vetores previamente persistidos em disco.
        """
        self.df = pd.read_csv(data_path)
        
        self.w2v_model = Word2Vec.load('data/word2vec.model')
        self.movie_vectors = np.load('data/word2vec_embeddings.npy')

    def _get_average_vector(self, text):
        """
        Calcula o vetor composto médio a partir de todas as palavras conhecidas em uma sinopse.
        """
        words = str(text).split()
        vectors = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
        if len(vectors) == 0:
            return np.zeros(100)
        return np.mean(vectors, axis=0)

    def search(self, query, top_k=5):
        """
        Efetua a busca linear comparando o vetor médio da query com os vetores dos filmes via cosseno.
        """
        start_time = time.time()
        query_vec = self._get_average_vector(query).reshape(1, -1)
        similarities = cosine_similarity(query_vec, self.movie_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        search_time = time.time() - start_time
        
        return self.df.iloc[top_indices], search_time